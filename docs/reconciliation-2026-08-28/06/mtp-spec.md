# reconcile(mtp-spec) — 06 invariant record

Files: `vllm/config/speculative.py`, `vllm/model_executor/models/qwen3_5_mtp.py`,
`vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/pp_spec_broadcast.py`,
`vllm/model_executor/layers/quantized_draft_embedding.py`, `tests/v1/spec_decode/*`
Resolved conflict: SILENT overlap — fork + upstream both touched `gpu_model_runner.py` and
`utils.py` (divergence-map:26). Auto-merged by git, 0 markers.

Invariant: PP>1 target + single-stage MTP draft must run on TPU: `SupportsPP` guard satisfied,
last-rank draft vocab embedding quantized (no huge fp16 resident copy), spec tokens cross PP stages
via the broadcast channel.

Decision: HYBRID — fork MTP/PP port layered on upstream MRV2 (4aab2b0ebe) / sleep-mode (479eeb32d2) churn.
Fork features verified present at HEAD: `pp_spec_broadcast.py` (108 lines), `quantized_draft_embedding.py`
(133), `is_tpu_compact`/`pp_spec_broadcast` refs in `gpu_model_runner.py` (delta ++280).

Evidence (HEAD): `git grep -nc "pp_spec_broadcast\|SupportsPP\|quantized_draft_embedding" HEAD --
vllm/v1/worker/gpu_model_runner.py` ≥ 1; upstream `d1922cb5` has zero (measured).

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Residual risk: region-level clobber of fork spec/PP hunk body by upstream churn not byte-verified →
deferred to 07/08 behavioral probe.
Full entry: task/evidence/06-conflicts/resolution-log.md §V2.
