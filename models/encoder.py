import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict

from .components.embeddings import PatchEmbedding, PositionalEncoding
from .components.attention import TransformerBlock, CrossVariableFusion, GatedPatchFusion


class ScalePatchTST(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, dropout)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        patches = self.patch_embed(x)
        num_patches = patches.size(2)
        pos_embed = self.pos_encoding(num_patches, patches.device)
        patches = patches + pos_embed.unsqueeze(0).unsqueeze(0)
        patches = rearrange(patches, 'b v p d -> (b v) p d')
        for layer in self.layers:
            patches = layer(patches)
        patches = rearrange(patches, '(b v) p d -> b v p d', b=B, v=V)
        return patches


class MultiScaleEncoder(nn.Module):
    """Shared PatchTST with per-series FFT scales (Eqs. 1-3).

    Use variable-averaged spectra, p=round(max(selected_f)/f), and adaptive-pool
    alignment to the original patch count. These rounding/alignment choices
    are unspecified in the paper. len(scales) sets the selected FFT-bin count.
    """
    def __init__(self, scales: List[int], patch_len: int, stride: int, d_model: int,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if not scales:
            raise ValueError('At least one FFT scale is required')
        self.scales = list(scales)
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.backbone = ScalePatchTST(patch_len, stride, d_model, n_heads,
                                      n_layers, d_ff, dropout)
        self.scale_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, d_model))
            for _ in self.scales
        ])

    @torch.no_grad()
    def dominant_steps(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch, S) integer downsampling steps, excluding the DC bin."""
        spectrum = torch.fft.rfft(x.detach().float(), dim=-1).abs().mean(dim=1)
        if spectrum.shape[-1] <= 1:
            return torch.ones(len(x), len(self.scales), dtype=torch.long,
                              device=x.device)
        amplitudes = spectrum[:, 1:]
        count = min(len(self.scales), amplitudes.shape[-1])
        frequencies = amplitudes.topk(count, dim=-1, sorted=True).indices + 1
        if count < len(self.scales):
            frequencies = torch.cat([
                frequencies,
                frequencies[:, :1].expand(-1, len(self.scales) - count),
            ], dim=-1)
        highest = frequencies.max(dim=-1, keepdim=True).values
        return (highest / frequencies.float()).round().long().clamp_(1, x.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, T = x.shape
        target_patches = max(1, (T - self.patch_len) // self.stride + 1)
        steps = self.dominant_steps(x)
        scale_features = []
        for i, mlp in enumerate(self.scale_mlps):
            aligned = x.new_zeros(B, V, target_patches, self.d_model)
            # Batch samples with equal downsampling steps.
            for step_tensor in steps[:, i].unique():
                step = int(step_tensor)
                indices = torch.where(steps[:, i] == step_tensor)[0]
                group = x.index_select(0, indices)
                # Average pooling gives floor(T / p) outputs (Eq. 1).
                if step > 1:
                    group = F.avg_pool1d(group, kernel_size=step, stride=step)
                if group.shape[-1] < self.patch_len:
                    group = F.pad(group, (0, self.patch_len - group.shape[-1]),
                                  mode='replicate')
                features = self.backbone(group)
                group_size = len(indices)
                features = rearrange(features, 'b v p d -> (b v) d p')
                features = F.adaptive_avg_pool1d(features, target_patches)
                features = rearrange(features, '(b v) d p -> b v p d',
                                     b=group_size, v=V)
                aligned = aligned.index_copy(0, indices, features)
            scale_features.append(mlp(aligned))
        return torch.stack(scale_features, dim=0).sum(dim=0)


class Stage1Encoder(nn.Module):
    def __init__(self, seq_len: int, n_vars: int, num_classes: int, scales: List[int] = None,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_classes = num_classes
        self.multi_scale = MultiScaleEncoder(scales, patch_len, stride, d_model, n_heads, n_layers, d_ff, dropout)
        self.cross_var = CrossVariableFusion(d_model, n_heads, dropout)
        self.gated_fusion = GatedPatchFusion(d_model, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_feat = self.multi_scale(x)
        cross_var_feat = self.cross_var(multi_scale_feat)
        return self.gated_fusion(cross_var_feat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        repr = self.encode(x)
        return {'logits': self.classifier(repr), 'repr': repr}

    def get_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> tuple:
        loss = F.cross_entropy(outputs['logits'], labels)
        return loss, {'cls_loss': loss.item()}
