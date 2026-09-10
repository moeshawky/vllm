import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock torch for testing without installing it
mock_torch = MagicMock()
mock_torch.__version__ = "2.3.0"
mock_torch.library = MagicMock()
mock_torch.library.Library = MagicMock()
mock_torch.library.infer_schema = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.cpp_extension'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torch.library'] = mock_torch.library
sys.modules['torch.distributed'] = MagicMock()
sys.modules['torch.distributed.distributed_c10d'] = MagicMock()
sys.modules['torch.profiler'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['vllm._version'] = MagicMock()

try:
    import vllm.envs
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    _HAS_IMPORTS = True
except Exception as e:
    _HAS_IMPORTS = False
    print(f"Failed to import vllm modules for trace tests: {e}")

@pytest.mark.skipif(not _HAS_IMPORTS, reason="Missing dependencies for importing KVCacheManager")
def test_trace_kv_cache_manager_enabled():
    with patch("vllm.envs.VLLM_TRACE_SCHEDULER", True), \
         patch("vllm.v1.core.kv_cache_manager.logger.warning") as mock_warning:

        coordinator = MagicMock()
        coordinator.get_num_blocks_to_allocate.return_value = 10
        block_pool = MagicMock()
        block_pool.get_num_free_blocks.return_value = 20
        block_pool.get_usage.return_value = 0.5

        kv_manager = KVCacheManager(
            coordinator=coordinator,
            block_pool=block_pool,
            empty_kv_cache_blocks=MagicMock()
        )

        request = MagicMock()
        request.request_id = "test_req_id"

        kv_manager.allocate_slots(request, 10, 0)

        mock_warning.assert_called_with(
            "SCHED-TRACE kv_pre_alloc free_before=%d kv_usage=%.4f",
            20, 0.5
        )

@pytest.mark.skipif(not _HAS_IMPORTS, reason="Missing dependencies for importing KVCacheManager")
def test_trace_kv_cache_manager_disabled():
    with patch("vllm.envs.VLLM_TRACE_SCHEDULER", False), \
         patch("vllm.v1.core.kv_cache_manager.logger.warning") as mock_warning:

        coordinator = MagicMock()
        coordinator.get_num_blocks_to_allocate.return_value = 10
        block_pool = MagicMock()
        block_pool.get_num_free_blocks.return_value = 20
        block_pool.get_usage.return_value = 0.5

        kv_manager = KVCacheManager(
            coordinator=coordinator,
            block_pool=block_pool,
            empty_kv_cache_blocks=MagicMock()
        )

        request = MagicMock()
        request.request_id = "test_req_id"
        kv_manager.allocate_slots(request, 10, 0)

        mock_warning.assert_not_called()
