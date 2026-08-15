//! Streaming LLM via llama.cpp (Qwen GGUF).
//!
//! `llama-cpp-2` exposes the low-level llama.cpp API, so generation is a
//! manual token loop: tokenize → decode → sample → emit text piece. Tokens are
//! streamed to the caller through the `on_token` callback (which returns
//! `false` to stop early, used for barge-in).
//!
//! Used on every target: on Android/Termux the binary is cross-compiled on the
//! host with the Android NDK (so the `llama-cpp-sys-2` NDK requirement is met),
//! rather than built on-device.

use std::num::NonZeroU32;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::config::LlmConfig;
use crate::traits::Llm;

const N_BATCH: u32 = 512;
const TOP_P: f32 = 0.9;
const SEED: u32 = 1234;

pub struct QwenLlm {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    max_tokens: u32,
    temperature: f32,
    system_prompt: String,
}

impl QwenLlm {
    pub fn new(cfg: &LlmConfig) -> Result<Self> {
        // Silence the native llama.cpp/ggml stderr logs (model load progress,
        // graph reservation, "Flash Attention", "Gated Delta Net", etc.).
        // Real failures still surface through the `Result` return values.
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        let backend = LlamaBackend::init().context("failed to init llama backend")?;
        // CPU-only for the MVP (no GPU offload).
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params)
            .with_context(|| format!("failed to load LLM model {}", cfg.model_path.display()))?;
        Ok(Self {
            backend,
            model,
            n_ctx: cfg.n_ctx,
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            system_prompt: cfg.system_prompt.clone(),
        })
    }

    /// Wrap the user message in the Qwen chat template with the system prompt.
    fn build_prompt(&self, user: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            self.system_prompt, user
        )
    }
}

impl Llm for QwenLlm {
    fn generate(&mut self, prompt: &str, on_token: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        let full = self.build_prompt(prompt);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(N_BATCH)
            // Use all available cores — llama.cpp's default can be low, which
            // makes generation on a phone unnecessarily slow.
            .with_n_threads(std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4))
            .with_n_threads_batch(std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("failed to create LLM context")?;

        let tokens = self
            .model
            .str_to_token(&full, AddBos::Always)
            .context("failed to tokenize prompt")?;
        let last_index = tokens.len() as i32 - 1;
        let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            batch
                .add(token, i, &[0], i == last_index)
                .context("failed to add prompt token to batch")?;
        }
        ctx.decode(&mut batch).context("initial decode failed")?;

        let mut n_cur = batch.n_tokens();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = LlamaSampler::chain_simple([
            // Order matters: filters/temperature operate on logits, `dist`
            // (softmax + selection) must come last.
            LlamaSampler::top_p(TOP_P, 1),
            LlamaSampler::temp(self.temperature),
            LlamaSampler::dist(SEED),
        ]);

        let mut generated = 0u32;
        while generated < self.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if token == self.model.token_eos() {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode token")?;
            generated += 1;
            if !on_token(&piece) {
                break; // early stop (barge-in)
            }
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("failed to add token to batch")?;
            n_cur += 1;
            ctx.decode(&mut batch).context("decode failed")?;
        }

        Ok(())
    }
}
