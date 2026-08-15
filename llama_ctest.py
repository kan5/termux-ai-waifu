#!/usr/bin/env python3
"""Prove the system libllama.so (Termux llama-cpp pkg) can run Qwen on-device.

ctypes smoke-test: load model -> tokenize -> decode -> sampler -> detokenize.
Builds structs by hand against $PREFIX/include/llama.h (llama.cpp 0.18.1).
Run on Termux:  python llama_ctest.py
Key API facts learned (llama.cpp >= ~0.14, new model/vocab split):
  - llama_model_load_from_file + llama_model_get_vocab
  - llama_init_from_model (not deprecated llama_new_context_with_model)
  - samplers are low-level: llama_sampler_apply(&cur_p) + llama_sampler_accept
  - llama_batch.logits is int8_t*; llama_batch_get_one leaves it NULL -> must
    allocate an int8 mask (1 = request logits for that token) yourself
  - struct layouts in this file match the system header exactly.
"""
import ctypes

LIB = "/data/data/com.termux/files/usr/lib/libllama.so"
MODEL = "/data/data/com.termux/files/home/termux-ai-waifu/models/Qwen3-0.6B-abliterated-q4_k_m.gguf"
PROMPT = "Привет! Напиши одно короткое предложение."
N_GEN = 12

lib = ctypes.CDLL(LIB)
lib.llama_backend_init.argtypes = [ctypes.c_int]
lib.llama_backend_init(0)

# ---------------- model params ----------------
class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]

lib.llama_model_default_params.restype = LlamaModelParams
params = lib.llama_model_default_params()
params.n_gpu_layers = 0

lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]
lib.llama_model_load_from_file.restype = ctypes.c_void_p
model = lib.llama_model_load_from_file(MODEL.encode(), params)
assert model, "model load failed"
print("model loaded OK")

lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
lib.llama_model_get_vocab.restype = ctypes.c_void_p
vocab = lib.llama_model_get_vocab(model)
assert vocab, "no vocab"

# ---------------- context params ----------------
class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        # enums (each 4 bytes)
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        # booleans together at the end
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]

lib.llama_context_default_params.restype = LlamaContextParams
cparams = lib.llama_context_default_params()
cparams.n_ctx = 512
cparams.n_threads = 4

lib.llama_init_from_model.argtypes = [ctypes.c_void_p, LlamaContextParams]
lib.llama_init_from_model.restype = ctypes.c_void_p
ctx = lib.llama_init_from_model(model, cparams)
assert ctx, "ctx init failed"
print("context OK")

# ---------------- tokenize ----------------
lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
lib.llama_vocab_n_tokens.restype = ctypes.c_int32
nv = lib.llama_vocab_n_tokens(vocab)
print("vocab tokens:", nv)

TOK = ctypes.c_int32
lib.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                               ctypes.POINTER(TOK), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
lib.llama_tokenize.restype = ctypes.c_int32
n_max = 512
tokens = (TOK * n_max)()
pb = PROMPT.encode()
n_tok = lib.llama_tokenize(vocab, pb, len(pb), tokens, n_max, True, False)
assert n_tok > 0, "tokenize failed"
print("tokenized:", n_tok, "tokens")

# ---------------- sampler chain ----------------
class LlamaSamplerChainParams(ctypes.Structure):
    _fields_ = [("no_perf", ctypes.c_bool)]

lib.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams
lib.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
lib.llama_sampler_chain_init.restype = ctypes.c_void_p
sp = lib.llama_sampler_chain_default_params()
chain = lib.llama_sampler_chain_init(sp)

lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
lib.llama_sampler_init_temp.restype = ctypes.c_void_p
lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
lib.llama_sampler_init_top_p.restype = ctypes.c_void_p
lib.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
lib.llama_sampler_init_dist.restype = ctypes.c_void_p
lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.llama_sampler_chain_add(chain, lib.llama_sampler_init_top_p(0.9, 1))
lib.llama_sampler_chain_add(chain, lib.llama_sampler_init_temp(0.7))
lib.llama_sampler_chain_add(chain, lib.llama_sampler_init_dist(1234))

# ---------------- batch / decode / logits ----------------
class LlamaBatch(ctypes.Structure):
    _fields_ = [("n_tokens", ctypes.c_int32),
                ("token", ctypes.POINTER(TOK)),
                ("embd", ctypes.POINTER(ctypes.c_float)),
                ("pos", ctypes.POINTER(ctypes.c_int32)),
                ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
                ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
                ("logits", ctypes.POINTER(ctypes.c_int8))]

class LlamaTokenData(ctypes.Structure):
    _fields_ = [("id", TOK), ("logit", ctypes.c_float), ("p", ctypes.c_float)]

class LlamaTokenDataArray(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(LlamaTokenData)),
                ("size", ctypes.c_size_t),
                ("selected", ctypes.c_int64),
                ("sorted", ctypes.c_bool)]

lib.llama_batch_get_one.argtypes = [ctypes.POINTER(TOK), ctypes.c_int32]
lib.llama_batch_get_one.restype = LlamaBatch
lib.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]
lib.llama_decode.restype = ctypes.c_int32
lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
lib.llama_sampler_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(LlamaTokenDataArray)]
lib.llama_sampler_accept.argtypes = [ctypes.c_void_p, TOK]
lib.llama_token_to_piece.argtypes = [ctypes.c_void_p, TOK, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
lib.llama_token_to_piece.restype = ctypes.c_int32
lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, TOK]
lib.llama_vocab_is_eog.restype = ctypes.c_bool

# token data array buffer (vocab sized)
data_arr = (LlamaTokenData * nv)()
cur_p = LlamaTokenDataArray(ctypes.cast(data_arr, ctypes.POINTER(LlamaTokenData)), nv, -1, False)

# ---------------- generate ----------------
batch = lib.llama_batch_get_one(tokens, n_tok)
# llama_batch_get_one leaves `logits` as NULL — allocate an int8 mask
# requesting logits only for the last token.
logits_mask = (ctypes.c_int8 * n_tok)(*([0] * (n_tok - 1) + [1]))
batch.logits = ctypes.cast(logits_mask, ctypes.POINTER(ctypes.c_int8))
out = []
for i in range(N_GEN):
    rc = lib.llama_decode(ctx, batch)
    assert rc == 0, f"decode failed rc={rc}"
    logits = lib.llama_get_logits_ith(ctx, batch.n_tokens - 1)
    # fill token_data_array from logits
    for j in range(nv):
        data_arr[j].id = j
        data_arr[j].logit = logits[j]
        data_arr[j].p = 0.0
    cur_p.size = nv
    cur_p.selected = -1
    cur_p.sorted = False
    lib.llama_sampler_apply(chain, ctypes.byref(cur_p))
    assert cur_p.selected >= 0, "sampler selected < 0"
    token = data_arr[cur_p.selected].id
    lib.llama_sampler_accept(chain, token)
    if lib.llama_vocab_is_eog(vocab, token):
        print("[EOG]")
        break
    buf = ctypes.create_string_buffer(64)
    ln = lib.llama_token_to_piece(vocab, token, buf, 64, 0, False)
    out.append(buf.raw[:ln].decode("utf-8", errors="replace"))
    one = (TOK * 1)(token)
    batch = lib.llama_batch_get_one(one, 1)
    lm = (ctypes.c_int8 * 1)(1)
    batch.logits = ctypes.cast(lm, ctypes.POINTER(ctypes.c_int8))

print("GENERATED:", "".join(out))
print("RESULT: PASS — system libllama.so generates text")
