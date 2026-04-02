# flash_attn_cpu/losses/cross_entropy.py
"""CPU-compatible cross entropy loss stub."""
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CPU fallback: standard PyTorch CrossEntropyLoss."""

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0,
                 inplace_backward=False, **kwargs):
        super().__init__(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
