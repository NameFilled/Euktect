# flash_attn_cpu/modules/mha.py
"""CPU-compatible Multi-Head Attention stub replacing flash_attn.modules.mha."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """CPU-compatible Multi-Head Attention stub for flash_attn.modules.mha.MHA."""

    def __init__(self, embed_dim, num_heads=None, causal=False, layer_idx=None,
                 bias=True, dropout=0.0, softmax_scale=None,
                 fused_bias_fc=False, dwconv=False, rotary_emb_dim=0,
                 rotary_emb_scale_base=0, rotary_emb_interleaved=False,
                 use_flash_attn=False, checkpointing=False, return_residual=False,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        if num_heads is None:
            num_heads = max(1, embed_dim // 64)
        self.num_heads = num_heads
        self.causal = causal
        self.return_residual = return_residual
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None,
                max_seqlen=None, mixer_subset=None, inference_params=None, **kwargs):
        B, L, D = x.shape
        head_dim = D // self.num_heads

        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=x.device, dtype=x.dtype),
                diagonal=1
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(attn_out)
        return out if not self.return_residual else (out, x)


# In CPU-only context, parallel is same as regular
ParallelMHA = MHA
