"""Tests for hybrid KVCache checkpoint-drop propagation.

Verifies the eagle_verified pollution fix and the 4-combo logic
table for both find_longest_cache_hit and per_group paths.
"""

import sys
import unittest.mock as mock


class _FakeTpuPlatform:
    """Stand-in for TpuInferencePlatform so vllm imports work."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def dispatch_key(self):
        return "XLA"

    def is_pin_memory_available(self):
        return False

    def __getattr__(self, name):
        return mock.MagicMock()


# vllm import chain pulls in tpu_inference; mock it so tests
# run without the full TPU stack installed.
_triton_mock = mock.MagicMock()
_triton_mock.__version__ = "3.0.0"
_triton_mock.__spec__ = mock.MagicMock()


def _mock_pkg() -> mock.MagicMock:
    """Mock that supports submodule imports (acts like a package)."""
    _m = mock.MagicMock()
    _m.__path__ = []
    return _m


class _ModuleMock(mock.MagicMock):
    """Mock that acts like a proper module for import machinery.

    Returns a new _ModuleMock for any attribute access so that
    submodule imports (e.g. transformers.feature_extraction_utils)
    work without pre-registering every leaf module.
    """

    __path__: list = []  # type: ignore[assignment]
    __spec__ = mock.MagicMock()

    def __getattr__(self, name: str) -> "_ModuleMock":  # type: ignore[override]
        if name.startswith("__") and name.endswith("__"):
            return mock.MagicMock()
        return _ModuleMock()


def _setup_mocks() -> None:
    """Pre-register mocks for modules missing from .venv."""
    _sys_mods: dict[str, mock.MagicMock] = {
        "tpu_inference": _mock_pkg(),
        "tpu_inference.platforms": mock.MagicMock(),
        "tpu_inference.layers": mock.MagicMock(),
        "tpu_inference.layers.common": mock.MagicMock(),
        "tpu_inference.layers.common.sharding": mock.MagicMock(),
        "tpu_inference.utils": mock.MagicMock(),
        "torch_xla": mock.MagicMock(),
        "torch_xla.core": mock.MagicMock(),
        "torch_xla.core.xla_model": mock.MagicMock(),
        "torch_xla._internal": mock.MagicMock(),
        "torch_xla._internal.tpu": mock.MagicMock(),
        "cpuinfo": mock.MagicMock(),
        "regex": mock.MagicMock(),
        "zmq": _mock_pkg(),
        "zmq.asyncio": mock.MagicMock(),
        "urllib3": _mock_pkg(),
        "urllib3.util": mock.MagicMock(),
        "pybase64": mock.MagicMock(),
        "msgspec": mock.MagicMock(),
        "transformers": _mock_pkg(),
        "transformers.utils": mock.MagicMock(),
        "transformers.feature_extraction_utils": mock.MagicMock(),
        "transformers.image_processing_utils": mock.MagicMock(),
        "transformers.configuration_utils": mock.MagicMock(),
        "transformers.processing_utils": mock.MagicMock(),
        "transformers.video_processing_utils": mock.MagicMock(),
        "transformers.models": _mock_pkg(),
        "transformers.models.auto": _mock_pkg(),
        "transformers.models.auto.image_processing_auto": mock.MagicMock(),
        "transformers.models.auto.modeling_auto": mock.MagicMock(),
        "transformers.models.auto.tokenization_auto": mock.MagicMock(),
        "transformers.models.auto.video_processing_auto": mock.MagicMock(),
        "huggingface_hub": _mock_pkg(),
        "huggingface_hub.utils": mock.MagicMock(),
        "huggingface_hub.constants": mock.MagicMock(),
        "huggingface_hub.errors": mock.MagicMock(),
        "safetensors": _mock_pkg(),
        "safetensors.torch": mock.MagicMock(),
        "cbor2": mock.MagicMock(),
        "importlib.metadata": mock.MagicMock(),
        "aiohttp": mock.MagicMock(),
        "requests": mock.MagicMock(),
        "cachetools": mock.MagicMock(),
        "cloudpickle": mock.MagicMock(),
        "uvloop": mock.MagicMock(),
        "prometheus_client": mock.MagicMock(),
        "llguidance": mock.MagicMock(),
        "xgrammar": mock.MagicMock(),
        "triton": _triton_mock,
        "openai_harmony": mock.MagicMock(),
        "openai": _mock_pkg(),
        "openai.types": _mock_pkg(),
        "openai.types.responses": _mock_pkg(),
        "openai.types.responses.response_function_tool_call_output_item": mock.MagicMock(),
    }
    for _name, _mod in _sys_mods.items():
        sys.modules[_name] = _mod
    sys.modules["tpu_inference.platforms"].TpuPlatform = _FakeTpuPlatform
    sys.modules["importlib.metadata"].version.return_value = "5.0.0"
    # msgspec.Struct must be a real class (not MagicMock) to avoid
    # metaclass conflict when SamplingParams subclasses it.
    class _MsgspecStruct:  # type: ignore[misc]
        """Mock for msgspec.Struct."""

        def __init_subclass__(cls, **kwargs: object) -> None:
            pass

    sys.modules["msgspec"].Struct = _MsgspecStruct
    sys.modules["msgspec"].msgpack = mock.MagicMock()


_setup_mocks()

import pytest
from unittest.mock import MagicMock

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    FullAttentionSpec,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator


def _setup_coord(use_eagle: bool, drop_override: bool = False
                 ) -> HybridKVCacheCoordinator:
    """Build a HybridKVCacheCoordinator with fa + 2 mamba groups."""
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16, 16),),
        dtypes=(None,)  # type: ignore[arg-type]
    )
    fa_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=None,  # type: ignore[arg-type]
    )
    kv_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["fa_layer"], kv_cache_spec=fa_spec
            ),
            KVCacheGroupSpec(
                layer_names=["mamba_1"], kv_cache_spec=mamba_spec
            ),
            KVCacheGroupSpec(
                layer_names=["mamba_2"], kv_cache_spec=mamba_spec
            ),
        ],
    )
    coord = HybridKVCacheCoordinator(
        kv_cache_config=kv_config,
        max_model_len=128,
        max_in_flight_tokens=128,
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    for mgr in coord.single_type_managers:
        mgr_cls = type(mgr)
        mgr_cls.find_longest_cache_hit = MagicMock(return_value=([[]], 0))
    return coord


@pytest.mark.parametrize(
    "use_eagle,manager_drop,expected_drop",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_hybrid_eagle_drop_logic_table(
    use_eagle: bool,
    manager_drop: bool,
    expected_drop: bool,
) -> None:
    """Logic table for drop_eagle_block in both coordinator paths.

    Four combos covering ordinary Eagle vs custom TPU drop flag.
    """
    coord = _setup_coord(use_eagle)
    mgr1 = coord.single_type_managers[1]
    mgr2 = coord.single_type_managers[2]
    mgr1.drop_eagle_checkpoint_block = manager_drop
    mgr2.drop_eagle_checkpoint_block = manager_drop

    mgr1_cls = type(mgr1)
    mgr1_cls.find_longest_cache_hit.reset_mock()
    coord.find_longest_cache_hit([], 16)
    kwargs = mgr1_cls.find_longest_cache_hit.call_args.kwargs
    assert kwargs["drop_eagle_block"] == expected_drop, (
        f"hit table: use_eagle={use_eagle} manager_drop={manager_drop}"
    )

    mgr1_cls.find_longest_cache_hit.reset_mock()
    coord.find_longest_cache_hit_per_group([], 16)
    kwargs = mgr1_cls.find_longest_cache_hit.call_args.kwargs
    assert kwargs["drop_eagle_block"] == expected_drop, (
        f"per_group table: use_eagle={use_eagle} "
        f"manager_drop={manager_drop}"
    )


def test_eagle_verified_suppression() -> None:
    """Custom drop flag preserves True even when eagle_verified fires.

    On the second while-loop pass, idx is in eagle_verified so
    traditional=False, but the custom TPU flag keeps drop=True.
    """
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16, 16),),
        dtypes=(None,),  # type: ignore[arg-type]
    )
    fa_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=None,  # type: ignore[arg-type]
    )
    kv_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["fa_layer"], kv_cache_spec=fa_spec
            ),
            KVCacheGroupSpec(
                layer_names=["mamba_layer"], kv_cache_spec=mamba_spec
            ),
        ],
    )
    coord = HybridKVCacheCoordinator(
        kv_cache_config=kv_config,
        max_model_len=128,
        max_in_flight_tokens=128,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    fa_manager = coord.single_type_managers[0]
    mamba_manager = coord.single_type_managers[1]
    mamba_manager.drop_eagle_checkpoint_block = True

    mamba_cls = type(mamba_manager)
    mamba_cls.find_longest_cache_hit = MagicMock(
        side_effect=[([[]], 16), ([[]], 16)]
    )

    coord.find_longest_cache_hit([], 16)
    last_kwargs = mamba_cls.find_longest_cache_hit.call_args.kwargs
    assert last_kwargs["drop_eagle_block"] is True
