import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x) + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x


class CrossVariableFusion(nn.Module):
    """Per-patch variable self-attention with the Eq. (6) residual."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.var_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, P, D = x.shape
        tokens = rearrange(x, 'b v p d -> (b p) v d')
        attended, _ = self.var_attn(tokens, tokens, tokens, need_weights=False)
        return rearrange(tokens + attended, '(b p) v d -> b v p d', b=B, p=P)


class GatedPatchFusion(nn.Module):
    """Eq. (7)-(9): unconstrained MLP scores, patch softmax, variable mean."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(x), dim=2)
        return (x * weights).sum(dim=2).mean(dim=1)
