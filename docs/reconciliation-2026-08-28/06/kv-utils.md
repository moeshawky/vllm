# reconcile(kv-utils) — 06 invariant record

Files: `vllm/v1/attention/backends/utils.py`, `vllm/v1/core/kv_cache_utils.py`
Resolved conflict: SILENT overlap — fork + upstream both touched `utils.py` (divergence-map:26).
Auto-merged, 0 markers.

Invariant: KV-cache layout is config-authoritative over any stale global — `get_kv_cache_layout` /
`set_kv_cache_layout` shims delegate to the 6-value state; `is_tpu_compact` admission bypass for
compact Mamba.

Decision: HYBRID (utils.py shims on upstream state) / PORT LOCAL (kv_cache_utils.py `is_tpu_compact`
absent upstream, grep 0 at `d1922cb5`).

Evidence (HEAD): `git grep -nc "get_kv_cache_layout\|set_kv_cache_layout" HEAD -- vllm/v1/attention/backends/utils.py`
= 10; `is_tpu_compact` refs in `kv_cache_utils.py` = 2.

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Residual risk: region-level clobber deferred to 07/08.
Full entry: task/evidence/06-conflicts/resolution-log.md §V1/V3.
