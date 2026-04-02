# flash_attn_cpu/ops/fused_dense.py
"""CPU-compatible fused dense stubs."""
import torch.nn as nn


class ColumnParallelLinear(nn.Linear):
    """CPU fallback: standard nn.Linear (no column-parallel needed on CPU)."""

    def __init__(self, in_features, out_features, process_group=None,
                 bias=True, sequence_parallel=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)


class FusedDense(nn.Linear):
    """CPU fallback: standard nn.Linear."""
    pass
