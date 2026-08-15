//! Native LLM backend via the system `libllama.so` (Termux `llama-cpp` pkg).
//!
//! On Android/Termux the `llama-cpp-2` crate cannot build (its build.rs requires
//! the Android NDK, which Termux lacks). Instead we dlopen the prebuilt system
//! `libllama.so` (llama.cpp 0.18.1) and drive it directly. The struct layouts
//! below are copied verbatim from `$PREFIX/include/llama.h` — a wrong layout
//! segfaults or trips `GGML_ASSERT(!ml.no_alloc)`.
//!
//! This module is compiled only on `target_os = "android"` (see `mod.rs`).

use std::ffi::{c_char, c_float, c_int, c_void, CString};

use anyhow::{Context, Result};
use libloading::Library;

use crate::config::LlmConfig;
use crate::traits::Llm;

type LlamaToken = i32;

// ---------- llama_model_params (16 fields) ----------
#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaModelParams {
    devices: *mut c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: c_int,
    split_mode: c_int,
    load_mode: c_int,
    main_gpu: c_int,
    tensor_split: *const c_float,
    progress_callback: *mut c_void,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

// ---------- llama_context_params (30 fields) ----------
#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_threads: c_int,
    n_threads_batch: c_int,
    // enums (4 bytes each)
    ctx_type: c_int,
    rope_scaling_type: c_int,
    pooling_type: c_int,
    attention_type: c_int,
    flash_attn_type: c_int,
    rope_freq_base: c_float,
    rope_freq_scale: c_float,
    yarn_ext_factor: c_float,
    yarn_attn_factor: c_float,
    yarn_beta_fast: c_float,
    yarn_beta_slow: c_float,
    yarn_orig_ctx: u32,
    defrag_thold: c_float,
    cb_eval: *mut c_void,
    cb_eval_user_data: *mut c_void,
    type_k: c_int,
    type_v: c_int,
    abort_callback: *mut c_void,
    abort_callback_data: *mut c_void,
    // booleans together at the end
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
    ctx_other: *mut c_void,
}

// ---------- llama_sampler_chain_params ----------
#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaSamplerChainParams {
    no_perf: bool,
}

// ---------- llama_batch (7 fields) ----------
#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaBatch {
    n_tokens: c_int,
    token: *mut LlamaToken,
    embd: *mut c_float,
    pos: *mut c_int,
    n_seq_id: *mut c_int,
    seq_id: *mut *mut c_int,
    logits: *mut i8,
}

// ---------- llama_token_data / _array ----------
#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaTokenData {
    id: LlamaToken,
    logit: c_float,
    p: c_float,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaTokenDataArray {
    data: *mut LlamaTokenData,
    size: usize,
    selected: i64,
    sorted: bool,
}

// ---------- function pointers (copy of `Symbol`s, keep Library alive) ----------
struct LlamaApi {
    backend_init: unsafe extern "C" fn(c_int),
    model_default_params: unsafe extern "C" fn() -> LlamaModelParams,
    model_load_from_file: unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut c_void,
    model_get_vocab: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    context_default_params: unsafe extern "C" fn() -> LlamaContextParams,
    init_from_model: unsafe extern "C" fn(*mut c_void, LlamaContextParams) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    vocab_n_tokens: unsafe extern "C" fn(*mut c_void) -> i32,
    tokenize: unsafe extern "C" fn(*mut c_void, *const c_char, i32, *mut LlamaToken, i32, bool, bool) -> i32,
    token_to_piece: unsafe extern "C" fn(*mut c_void, LlamaToken, *mut c_char, i32, i32, bool) -> i32,
    vocab_is_eog: unsafe extern "C" fn(*mut c_void, LlamaToken) -> bool,
    sampler_chain_default_params: unsafe extern "C" fn() -> LlamaSamplerChainParams,
    sampler_chain_init: unsafe extern "C" fn(LlamaSamplerChainParams) -> *mut c_void,
    sampler_chain_add: unsafe extern "C" fn(*mut c_void, *mut c_void),
    sampler_init_temp: unsafe extern "C" fn(c_float) -> *mut c_void,
    sampler_init_top_p: unsafe extern "C" fn(c_float, usize) -> *mut c_void,
    sampler_init_dist: unsafe extern "C" fn(u32) -> *mut c_void,
    batch_get_one: unsafe extern "C" fn(*mut LlamaToken, i32) -> LlamaBatch,
    decode: unsafe extern "C" fn(*mut c_void, LlamaBatch) -> i32,
    get_logits_ith: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_float,
    sampler_apply: unsafe extern "C" fn(*mut c_void, *mut LlamaTokenDataArray),
    sampler_accept: unsafe extern "C" fn(*mut c_void, LlamaToken),
}

impl LlamaApi {
    unsafe fn load(lib: &Library) -> Result<Self> {
        macro_rules! sym {
            ($name:literal, $t:ty) => {
                *lib.get::<$t>($name).with_context(|| format!("missing symbol {}", String::from_utf8_lossy($name)))?
            };
        }
        Ok(Self {
            backend_init: sym!(b"llama_backend_init", unsafe extern "C" fn(c_int)),
            model_default_params: sym!(b"llama_model_default_params", unsafe extern "C" fn() -> LlamaModelParams),
            model_load_from_file: sym!(b"llama_model_load_from_file", unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut c_void),
            model_get_vocab: sym!(b"llama_model_get_vocab", unsafe extern "C" fn(*mut c_void) -> *mut c_void),
            context_default_params: sym!(b"llama_context_default_params", unsafe extern "C" fn() -> LlamaContextParams),
            init_from_model: sym!(b"llama_init_from_model", unsafe extern "C" fn(*mut c_void, LlamaContextParams) -> *mut c_void),
            free: sym!(b"llama_free", unsafe extern "C" fn(*mut c_void)),
            vocab_n_tokens: sym!(b"llama_vocab_n_tokens", unsafe extern "C" fn(*mut c_void) -> i32),
            tokenize: sym!(b"llama_tokenize", unsafe extern "C" fn(*mut c_void, *const c_char, i32, *mut LlamaToken, i32, bool, bool) -> i32),
            token_to_piece: sym!(b"llama_token_to_piece", unsafe extern "C" fn(*mut c_void, LlamaToken, *mut c_char, i32, i32, bool) -> i32),
            vocab_is_eog: sym!(b"llama_vocab_is_eog", unsafe extern "C" fn(*mut c_void, LlamaToken) -> bool),
            sampler_chain_default_params: sym!(b"llama_sampler_chain_default_params", unsafe extern "C" fn() -> LlamaSamplerChainParams),
            sampler_chain_init: sym!(b"llama_sampler_chain_init", unsafe extern "C" fn(LlamaSamplerChainParams) -> *mut c_void),
            sampler_chain_add: sym!(b"llama_sampler_chain_add", unsafe extern "C" fn(*mut c_void, *mut c_void)),
            sampler_init_temp: sym!(b"llama_sampler_init_temp", unsafe extern "C" fn(c_float) -> *mut c_void),
            sampler_init_top_p: sym!(b"llama_sampler_init_top_p", unsafe extern "C" fn(c_float, usize) -> *mut c_void),
            sampler_init_dist: sym!(b"llama_sampler_init_dist", unsafe extern "C" fn(u32) -> *mut c_void),
            batch_get_one: sym!(b"llama_batch_get_one", unsafe extern "C" fn(*mut LlamaToken, i32) -> LlamaBatch),
            decode: sym!(b"llama_decode", unsafe extern "C" fn(*mut c_void, LlamaBatch) -> i32),
            get_logits_ith: sym!(b"llama_get_logits_ith", unsafe extern "C" fn(*mut c_void, i32) -> *mut c_float),
            sampler_apply: sym!(b"llama_sampler_apply", unsafe extern "C" fn(*mut c_void, *mut LlamaTokenDataArray)),
            sampler_accept: sym!(b"llama_sampler_accept", unsafe extern "C" fn(*mut c_void, LlamaToken)),
        })
    }
}

const N_BATCH: u32 = 512;
const TOP_P: f32 = 0.9;
const SEED: u32 = 1234;
// Typical Termux prefix; overridable via env.
const DEFAULT_LIB: &str = "/data/data/com.termux/files/usr/lib/libllama.so";

pub struct NativeLlm {
    _lib: Library, // must outlive `api` (kept alive for the struct's lifetime)
    api: LlamaApi,
    model: *mut c_void,
    vocab: *mut c_void,
    n_ctx: u32,
    max_tokens: u32,
    temperature: f32,
    system_prompt: String,
}

// Raw pointers are not Send by default; each instance is used from a single
// blocking task, so it is safe to move it across threads once.
unsafe impl Send for NativeLlm {}

impl NativeLlm {
    pub fn new(cfg: &LlmConfig) -> Result<Self> {
        let lib_path = std::env::var("LLAMA_LIB_PATH").unwrap_or_else(|_| DEFAULT_LIB.to_string());
        // SAFETY: the Library stays alive for the struct's lifetime.
        let lib = unsafe { Library::new(&lib_path) }
            .with_context(|| format!("failed to dlopen {lib_path}"))?;
        let api = unsafe { LlamaApi::load(&lib) }?;

        unsafe {
            api.backend_init(0);

            let mut params = api.model_default_params();
            params.n_gpu_layers = 0;
            // (no use_mmap field in llama.cpp 0.18.1; mmap is the default)
            let path = CString::new(cfg.model_path.to_string_lossy().as_bytes())
                .context("model path contains NUL")?;
            let model = api.model_load_from_file(path.as_ptr(), params);
            if model.is_null() {
                anyhow::bail!("llama_model_load_from_file returned null");
            }
            let vocab = api.model_get_vocab(model);
            if vocab.is_null() {
                anyhow::bail!("llama_model_get_vocab returned null");
            }

            Ok(Self {
                _lib: lib,
                api,
                model,
                vocab,
                n_ctx: cfg.n_ctx,
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                system_prompt: cfg.system_prompt.clone(),
            })
        }
    }

    fn build_prompt(&self, user: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            self.system_prompt, user
        )
    }
}

impl Drop for NativeLlm {
    fn drop(&mut self) {
        unsafe { self.api.free(self.model) }
    }
}

impl Llm for NativeLlm {
    fn generate(&mut self, prompt: &str, on_token: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        let full = self.build_prompt(prompt);

        unsafe {
            // --- context ---
            let mut cparams = self.api.context_default_params();
            cparams.n_ctx = self.n_ctx;
            cparams.n_batch = N_BATCH;
            cparams.n_threads = 4;
            let ctx = self.api.init_from_model(self.model, cparams);
            if ctx.is_null() {
                anyhow::bail!("llama_init_from_model returned null");
            }

            // --- tokenize ---
            let text = CString::new(full.as_bytes()).context("prompt contains NUL")?;
            let mut tokens = vec![0i32; self.n_ctx as usize];
            let n_tok = self.api.tokenize(
                self.vocab,
                text.as_ptr(),
                full.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                false,
            );
            if n_tok <= 0 {
                self.api.free(ctx);
                anyhow::bail!("llama_tokenize failed");
            }
            tokens.truncate(n_tok as usize);

            // --- initial batch: request logits for the last token ---
            let mut batch = self.api.batch_get_one(tokens.as_mut_ptr(), n_tok);
            let mut logits_mask = vec![0i8; n_tok as usize];
            logits_mask[n_tok as usize - 1] = 1;
            batch.logits = logits_mask.as_mut_ptr();
            let rc = self.api.decode(ctx, batch);
            if rc != 0 {
                self.api.free(ctx);
                anyhow::bail!("initial decode failed rc={rc}");
            }
            let mut n_cur = n_tok as i32;

            // --- sampler chain (dist must be last) ---
            let sp = self.api.sampler_chain_default_params();
            let chain = self.api.sampler_chain_init(sp);
            let top_p = self.api.sampler_init_top_p(TOP_P, 1);
            let temp = self.api.sampler_init_temp(self.temperature);
            let dist = self.api.sampler_init_dist(SEED);
            self.api.sampler_chain_add(chain, top_p);
            self.api.sampler_chain_add(chain, temp);
            self.api.sampler_chain_add(chain, dist);

            // --- generation loop ---
            let n_vocab = self.api.vocab_n_tokens(self.vocab) as usize;
            let mut token_data = vec![
                LlamaTokenData { id: 0, logit: 0.0, p: 0.0 };
                n_vocab
            ];
            let mut cur_p = LlamaTokenDataArray {
                data: token_data.as_mut_ptr(),
                size: 0,
                selected: -1,
                sorted: false,
            };

            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut piece_buf = [0u8; 256];
            let mut generated = 0u32;

            while generated < self.max_tokens {
                let logits = self.api.get_logits_ith(ctx, n_cur - 1);
                if logits.is_null() {
                    self.api.free(ctx);
                    anyhow::bail!("llama_get_logits_ith returned null");
                }
                for j in 0..n_vocab {
                    token_data[j].id = j as i32;
                    token_data[j].logit = *logits.add(j);
                    token_data[j].p = 0.0;
                }
                cur_p.size = n_vocab;
                cur_p.selected = -1;
                cur_p.sorted = false;

                self.api.sampler_apply(chain, &mut cur_p);
                if cur_p.selected < 0 {
                    self.api.free(ctx);
                    anyhow::bail!("sampler selected < 0");
                }
                let token = token_data[cur_p.selected as usize].id;
                self.api.sampler_accept(chain, token);

                if self.api.vocab_is_eog(self.vocab, token) {
                    break;
                }

                let n = self.api.token_to_piece(
                    self.vocab,
                    token,
                    piece_buf.as_mut_ptr() as *mut c_char,
                    piece_buf.len() as i32,
                    0,
                    false,
                );
                let bytes = &piece_buf[..n.max(0) as usize];
                let mut out = String::new();
                let (_, _) = decoder.decode_to_string(bytes, &mut out, false);
                generated += 1;
                if !on_token(&out) {
                    break; // barge-in
                }

                // advance batch to the single new token
                let mut one = [token];
                batch = self.api.batch_get_one(one.as_mut_ptr(), 1);
                let mut lm = [1i8];
                batch.logits = lm.as_mut_ptr();
                n_cur += 1;
                let rc = self.api.decode(ctx, batch);
                if rc != 0 {
                    self.api.free(ctx);
                    anyhow::bail!("decode failed rc={rc}");
                }
            }

            // flush decoder tail (rare; emits any trailing partial multibyte char)
            let mut tail = String::new();
            let _ = decoder.decode_to_string(&[], &mut tail, true);
            if !tail.is_empty() {
                let _ = on_token(&tail);
            }

            self.api.free(ctx);
            Ok(())
        }
    }
}
