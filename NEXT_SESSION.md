# Next Session — qwen38-flash-next on TPU v5e-8

**Pushed:** 2026-08-30 01:17 UTC heads
- vllm-src `ecb4462` (main) — fix HC mix unpack 3→2
- tpu-inference `bcce252` (main) — feat fp8 bank-direct offload 512E S=32

**Model:** /kaggle/input/models/manarmilad/qwen3.8-flash-next-uncensored/transformers/fp8/1
**Banks:** /kaggle/input/datasets/moeshawky/qwen38-flash-expert-banks (48× layer_*.bank 113G, RO)

**What worked**
- 374s shard load (131 shards, NFS) + 48 banks (gate/up/down f8 + scales bf16 → w13 [512,1280,2560] w2 [512,2560,640])
- host admission 196GiB avail, 3.53G/layer host, S=32 slots

**Bugs fixed tonight**
- 00:41:15 `ValueError not enough values to unpack (expected 3, got 2)` at common/model.py:355 attn_hc.mix — fixed to `block_input, injection = attn_hc.mix`
- 01:14:25 `TracerArrayConversionError bfloat16[16,512]` at interface/moe.py:174 host routing inside jit — requires `--enforce-eager` (tpu_runner.py:1721 jax.disable_jit)

**Current launcher (canonical):** `serve_flash_tp4_bank5.sh` — TP4 b16 8192 ctx 32 seqs 0.70 util prefix+align --language-model-only --enforce-eager
```bash
export QWEN_BANK_DIR=/kaggle/input/datasets/moeshawky/qwen38-flash-expert-banks
export MOE_EXPERT_OFFLOAD=1 MOE_EXPERT_OFFLOAD_SLOTS=32
kaggle-backend run vllm-tpu -- vllm serve $MODEL --tensor_parallel_size 4 --dtype bfloat16 --max_model_len 8192 --max_num_seqs 32 --max_num_batched_tokens 16384 --gpu_memory_utilization 0.70 --enable_prefix_caching --mamba_cache_mode align --enforce-eager --reasoning_parser qwen3 --enable_auto_tool_choice --tool_call_parser qwen3_coder --trust-remote-code --language-model-only --port 8000 --served-model-name qwen38-flash
```

**Resume tomorrow**
```bash
git -C /kaggle/working/vllm-src pull
git -C /kaggle/working/tpu-inference pull
bash /kaggle/working/vllm-src/serve_flash_tp4_bank5.sh   # or tpu-inference copy
# watch
tail -f /tmp/vllm-server-bank5.log
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash","messages":[{"role":"user","content":"Say hello in 5 words"}],"max_tokens":64}'
```

**Expected:** ~6min shard load + ~15min banks (30s/2 layers) + eager warmup (no JIT) → Uvicorn within 20-25min wall. No chunked_mm error with --language-model-only.

**Files modified:**
- vllm-src: vllm/models/qwen4_exp/common/model.py, vllm/models/qwen4_exp/amd/model.py
- tpu-inference: tpu_inference/layers/vllm/quantization/fp8.py
