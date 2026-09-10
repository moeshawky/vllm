import sys
import unittest.mock as mock
import pytest

sys.modules['transformers'] = mock.MagicMock()
sys.modules['huggingface_hub'] = mock.MagicMock()
sys.modules['safetensors'] = mock.MagicMock()
sys.modules['safetensors.torch'] = mock.MagicMock()

import torch
from unittest.mock import MagicMock
from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheConfig,
    KVCacheGroupSpec
)

def test_hybrid_eagle_drop_logic_table():
    """
    Test the complete logic table for drop_eagle_block propagation in
    HybridKVCacheCoordinator logic for both single group and multi group.

    Table:
    use_eagle=False, manager_drop_override=False -> no drop
    use_eagle=True,  manager_drop_override=False -> drop
    use_eagle=False, manager_drop_override=True  -> drop
    use_eagle=True,  manager_drop_override=True  -> drop
    """
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16, 16),),
        dtypes=(torch.float32,)
    )
    fa_spec = FullAttentionSpec(
        block_size=16,
        num_heads=1,
        head_size=16,
        dtype=torch.float32
    )

    # We will instantiate HybridKVCacheCoordinator with differing use_eagle flags
    # and manually toggle the custom manager flag on the Mamba layers

    def setup_coord(use_eagle):
        kv_config = KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["fa_layer"], kv_cache_spec=fa_spec),
                KVCacheGroupSpec(layer_names=["mamba_layer1"], kv_cache_spec=mamba_spec),
                KVCacheGroupSpec(layer_names=["mamba_layer2"], kv_cache_spec=mamba_spec),
            ]
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
            hash_block_size=16
        )
        for manager in coord.single_type_managers:
            manager.find_longest_cache_hit = MagicMock(return_value=([[]], 0))
        return coord

    for use_eagle in [False, True]:
        for manager_drop in [False, True]:
            coord = setup_coord(use_eagle)

            # fa_manager group is 0, mamba_manager1 is 1, mamba_manager2 is 2
            # Mamba managers are grouped into one attention group by the coordinator.
            mamba_manager1 = coord.single_type_managers[1]
            mamba_manager2 = coord.single_type_managers[2]

            mamba_manager1.drop_eagle_checkpoint_block = manager_drop
            mamba_manager2.drop_eagle_checkpoint_block = manager_drop

            expected_drop = use_eagle or manager_drop

            # Reset mocks
            mamba_manager1.find_longest_cache_hit.reset_mock()

            coord.find_longest_cache_hit([], 16)

            # Find the call for the mamba spec group.
            # Hybrid groups combine identical specs into one group ID list: [1, 2]
            # It uses manager_cls (MambaManager) of the first manager in the group.
            # Since the manager mock is bound per instance, and MambaManager
            # find_longest_cache_hit is a classmethod, we'll check it from the class level or the bound method.
            # In our mock setup, the instance method is called.

            # Since both mamba_layer1 and mamba_layer2 share a spec group and manager class,
            # only the manager_cls.find_longest_cache_hit is called.
            kwargs = mamba_manager1.find_longest_cache_hit.call_args.kwargs
            assert kwargs['drop_eagle_block'] == expected_drop, (
                f"Failed hit table: use_eagle={use_eagle}, manager_drop={manager_drop}"
            )

            mamba_manager1.find_longest_cache_hit.reset_mock()
            coord.find_longest_cache_hit_per_group([], 16)
            kwargs = mamba_manager1.find_longest_cache_hit.call_args.kwargs
            assert kwargs['drop_eagle_block'] == expected_drop, (
                f"Failed hit_per_group table: use_eagle={use_eagle}, manager_drop={manager_drop}"
            )

if __name__ == "__main__":
    pytest.main([__file__])
def test_eagle_verified_suppression():
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16, 16),),
        dtypes=(torch.float32,)
    )
    fa_spec = FullAttentionSpec(
        block_size=16,
        num_heads=1,
        head_size=16,
        dtype=torch.float32
    )

    kv_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["fa_layer"], kv_cache_spec=fa_spec),
            KVCacheGroupSpec(layer_names=["mamba_layer"], kv_cache_spec=mamba_spec),
        ]
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
        hash_block_size=16
    )

    fa_manager = coord.single_type_managers[0]
    mamba_manager = coord.single_type_managers[1]

    # We simulate a case where eagle_verified suppression would typically set drop_eagle_block=False
    # but the custom tpu flag should force it to remain True.
    mamba_manager.drop_eagle_checkpoint_block = True

    # Force find_longest_cache_hit to return identical values so eagle_verified triggers
    fa_manager.find_longest_cache_hit = MagicMock(side_effect=[([[]], 16), ([[]], 16)])
    mamba_manager.find_longest_cache_hit = MagicMock(side_effect=[([[]], 16), ([[]], 16)])

    coord.find_longest_cache_hit([], 16)

    # On the second pass of the while True loop, `idx not in eagle_verified` is False,
    # meaning standard eagle would not drop. But the TPU flag should still keep it True.
    # The last call should have drop_eagle_block=True

    last_mamba_kwargs = mamba_manager.find_longest_cache_hit.call_args.kwargs
    assert last_mamba_kwargs['drop_eagle_block'] is True
