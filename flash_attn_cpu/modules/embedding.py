# flash_attn_cpu/modules/embedding.py
"""CPU-compatible embedding stub replacing flash_attn.modules.embedding."""
import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    """CPU-compatible GPT2Embeddings."""

    def __init__(self, embed_dim, vocab_size, max_position_embeddings=0,
                 padding_idx=None, word_embed_proj_dim=None, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            word_embed_proj_dim = embed_dim
        self.word_embeddings = nn.Embedding(
            vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs
        )
        self.project_in = (
            nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
            if word_embed_proj_dim != embed_dim else None
        )
        self.max_position_embeddings = max_position_embeddings
        if max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings


# CPU-only: parallel version same as regular
ParallelGPT2Embeddings = GPT2Embeddings
