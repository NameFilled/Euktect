# flash_attn_cpu/modules/mlp.py
"""CPU-compatible MLP stub replacing flash_attn.modules.mlp."""
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """CPU-compatible Mlp, mirroring flash_attn.modules.mlp.Mlp interface."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=F.gelu, return_residual=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


# FusedMLP and ParallelFusedMLP degenerate to Mlp on CPU
FusedMLP = Mlp
ParallelFusedMLP = Mlp
