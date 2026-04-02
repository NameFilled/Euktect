# flash_attn_cpu/ops/layer_norm.py
"""CPU-compatible layer norm stub."""
import torch.nn.functional as F


def dropout_add_layer_norm(x0, residual, weight, bias, dropout_p, epsilon,
                            rowscale=None, layerscale=None, prenorm=False,
                            x0_dtype=None, return_dropout_mask=False):
    """CPU fallback: unfused dropout + add + layer norm."""
    x = F.dropout(x0.float(), p=dropout_p, training=False)
    if residual is not None:
        x = x + residual.float()
    out = F.layer_norm(x, weight.shape, weight.float(), bias.float(), epsilon)
    out = out.to(x0.dtype)
    if prenorm:
        return out, x
    return out
