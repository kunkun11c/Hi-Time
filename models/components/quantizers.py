"""Residual K-Means targets fitted on training representations (Sec. 4.2)."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, ema_decay: float = 0.99,
                 eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        # ema_decay/eps are retained for config compatibility; fitting uses Lloyd.
        self.register_buffer('codebook', torch.zeros(codebook_size, dim))
        self.register_buffer('initted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, data: torch.Tensor, max_iter: int = 100, seed: int = 0):
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits

        if data.ndim != 2 or data.shape[1] != self.dim or len(data) == 0:
            raise ValueError(f'Expected nonempty (N, {self.dim}) training features')
        k = min(self.codebook_size, len(data))
        with threadpool_limits(limits=1):
            estimator = KMeans(n_clusters=k, init='k-means++', n_init=10,
                               max_iter=max(1, max_iter), random_state=seed)
            estimator.fit(data.detach().float().cpu().numpy())
        centers = torch.as_tensor(estimator.cluster_centers_,
                                  dtype=self.codebook.dtype,
                                  device=self.codebook.device)
        # Repeat fitted centers when K exceeds the training sample count.
        self.codebook.copy_(centers[torch.arange(self.codebook_size,
                                               device=centers.device) % k])
        self.initted.fill_(True)
        return self

    def init_codebook(self, data: torch.Tensor):
        """Fit the codebook if it is uninitialized."""
        if not self.initted:
            self.fit(data, seed=torch.initial_seed() % (2**32))

    def forward(self, x: torch.Tensor, code_embedding: nn.Embedding = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.initted:
            raise RuntimeError('Fit the K-Means codebook on training data first')
        indices = torch.cdist(x, self.codebook).argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        return quantized, indices, x - quantized


class ResidualQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 512, num_layers: int = 3,
                 ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.quantizers = nn.ModuleList([
            KMeansQuantizer(dim, codebook_size, ema_decay)
            for _ in range(num_layers)
        ])
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_layers)
        ])
        self.code_embeddings.requires_grad_(False)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, max_iter: int = 100, seed: int = 0):
        residual = x.detach()
        for i, quantizer in enumerate(self.quantizers):
            quantizer.fit(residual, max_iter=max_iter, seed=(seed + i) % (2**32))
            self.code_embeddings[i].weight.copy_(quantizer.codebook)
            _, _, residual = quantizer(residual)
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        codes, residuals, code_embeds = [], [], []
        residual = x
        quantized_sum = torch.zeros_like(x)
        for i, quantizer in enumerate(self.quantizers):
            quantized, indices, residual = quantizer(residual)
            codes.append(indices)
            residuals.append(residual)
            code_embeds.append(self.code_embeddings[i](indices))
            quantized_sum = quantized_sum + quantized
        return {'codes': codes, 'quantized': quantized_sum,
                'residuals': residuals, 'code_embeds': code_embeds}

    def get_code_embeddings(self, codes: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.code_embeddings[i](code) for i, code in enumerate(codes)]

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        if len(codes) != self.num_layers:
            raise ValueError('One code tensor is required per residual layer')
        return sum(F.embedding(code, quantizer.codebook)
                   for code, quantizer in zip(codes, self.quantizers))
