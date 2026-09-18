# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hybrid Multi-head Context (MHC) kernel wrappers."""

import torch

# this import will also register the custom ops
# import vllm.model_executor.kernels.mhc  # noqa: F401
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang


def _has_tilelang_mhc() -> bool:
    """Check whether TileLang MHC is available and supported on the current
    platform, excluding gfx942 ROCm.
    """
    if not has_tilelang():
        return False
    if current_platform.is_cuda():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx942

        # TileLang MHC currently produces incorrect results on gfx942. Keep
        # gfx942 on the existing torch/triton fallbacks until that path is fixed.
        return not on_gfx942()
    return False


HAS_TILELANG_MHC = _has_tilelang_mhc()  # Module-level boolean: TileLang MHC
    # available and supported on the current platform.
HAS_AITER_MHC = is_aiter_found_and_supported()  # Module-level boolean: AITER MHC
    # found and supported.


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        """Return True to indicate the MHC pre-op is always enabled."""
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MHC pre block on CUDA.

        Args:
            residual: HC residual tensor.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (post_mix, comb_mix, layer_input).
        """
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MHC pre block on HIP.

        Args:
            residual: HC residual tensor.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (post_mix, comb_mix, layer_input).
        """
        hidden_size = residual.shape[-1]
        hc_mult = residual.shape[-2]
        if HAS_AITER_MHC and hidden_size % 256 == 0 and hc_mult == 4:
            return torch.ops.vllm.mhc_pre_aiter(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        elif HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        else:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MHC pre block using the native torch fallback.

        Args:
            residual: HC residual tensor.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (post_mix, comb_mix, layer_input).
        """
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    def forward_xpu(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MHC pre block on XPU.

        Args:
            residual: HC residual tensor.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (post_mix, comb_mix, layer_input).
        """
        return torch.ops._xpu_C.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        """Return True to indicate the MHC post-op is always enabled."""
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MHC post block on CUDA.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.

        Returns:
            Combined output tensor.
        """
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MHC post block on HIP.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.

        Returns:
            Combined output tensor.
        """
        hidden_size = residual.shape[-1]
        hc_mult = residual.shape[-2]
        if HAS_AITER_MHC and hidden_size % 256 == 0 and hc_mult == 4:
            return torch.ops.vllm.mhc_post_aiter(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )
        if HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )
        else:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MHC post block using the native torch fallback.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.

        Returns:
            Combined output tensor.
        """
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MHC post block on XPU.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.

        Returns:
            Combined output tensor.
        """
        return torch.ops._xpu_C.mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        """Return True to indicate the HC head is always enabled."""
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Run the HC head fused kernel on CUDA.

        Args:
            hidden_states: Input hidden states tensor.
            hc_fn: HC function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_norm_eps: RMS normalization epsilon.
            hc_eps: HC head epsilon.

        Returns:
            Output tensor with hc_mult streams collapsed to one.
        """
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Run the HC head fused kernel on HIP.

        Args:
            hidden_states: Input hidden states tensor.
            hc_fn: HC function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_norm_eps: RMS normalization epsilon.
            hc_eps: HC head epsilon.

        Returns:
            Output tensor with hc_mult streams collapsed to one.
        """
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)

        if HAS_TILELANG_MHC:
            out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        else:
            num_tokens = hs_flat.shape[0]
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )

        return out.view(*outer_shape, hidden_size)

    def forward_native(self, *args, **kwargs):
        """Raise NotImplementedError: native hc_head implementation is unavailable.

        Raises:
            NotImplementedError: hc_head has no native implementation.
        """
        raise NotImplementedError("Native implementation of hc_head is not available")

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Run the HC head fused kernel on XPU.

        Args:
            hidden_states: Input hidden states tensor.
            hc_fn: HC function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_norm_eps: RMS normalization epsilon.
            hc_eps: HC head epsilon.

        Returns:
            Output tensor with hc_mult streams collapsed to one.
        """
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        torch.ops._xpu_C.hc_head_fused(
            hs_flat, hc_fn, hc_scale, hc_base, out, rms_norm_eps, hc_eps
        )
        return out.view(*outer_shape, hidden_size)


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        """Return True to indicate the fused MHC post-pre is always enabled."""
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fused MHC post-pre block on CUDA.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            tile_n: Tile dimension.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
        """
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fused MHC post-pre block on HIP.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            tile_n: Tile dimension.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
        """
        hidden_size = residual.shape[-1]
        hc_mult = residual.shape[-2]
        if HAS_AITER_MHC and hidden_size % 256 == 0 and hc_mult == 4:
            return torch.ops.vllm.mhc_fused_post_pre_aiter(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )
        if HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )
        return self.forward_native(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fused MHC post-pre block using native torch fallback.

        Decomposes into separate post and pre operations.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            tile_n: Tile dimension.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
        """
        # Decompose into post + pre (no fused kernel available).
        residual_cur = mhc_kernels.mhc_post_torch(
            x, residual, post_layer_mix, comb_res_mix
        )
        post_mix_cur, comb_mix_cur, layer_input_cur = mhc_kernels.mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fused MHC post-pre block on XPU.

        Args:
            x: Layer output tensor.
            residual: HC residual tensor.
            post_layer_mix: Post-layer mixing weights.
            comb_res_mix: Combined residual mixing weights.
            fn: Function tensor.
            hc_scale: HC scale tensor.
            hc_base: HC base tensor.
            rms_eps: RMS normalization epsilon.
            hc_pre_eps: MHC pre epsilon.
            hc_sinkhorn_eps: Sinkhorn epsilon.
            hc_post_mult_value: Post-multiply value.
            sinkhorn_repeat: Number of Sinkhorn iterations.
            n_splits: Number of splits.
            tile_n: Tile dimension.
            norm_weight: Optional normalization weights.
            norm_eps: Normalization epsilon.

        Returns:
            Tuple of (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
        """
        return torch.ops._xpu_C.mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
