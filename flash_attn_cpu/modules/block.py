# flash_attn_cpu/modules/block.py
"""CPU-compatible Block stub replacing flash_attn.modules.block."""
import torch
import torch.nn as nn
from functools import partial

try:
    from timm.models.layers import DropPath as StochasticDepth
except ImportError:
    from timm.layers import DropPath as StochasticDepth

from flash_attn_cpu.modules.mha import MHA
from flash_attn_cpu.modules.mlp import Mlp


class Block(nn.Module):
    """CPU-compatible Block, mirroring flash_attn.modules.block.Block interface."""

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., return_residual=False,
                 residual_in_fp32=False, fused_dropout_add_ln=False, **kwargs):
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=max(1, dim // 64))
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1)
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2)
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            mixer_kwargs = mixer_kwargs or {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = dropped + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_kwargs = mixer_kwargs or {}
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            hidden_states = self.norm1(hidden_states)
            if not isinstance(self.mlp, nn.Identity):
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.norm2(hidden_states)
            return hidden_states
