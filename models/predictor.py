"""GPT-2 predicts hierarchical codes and the class label (Eqs. 13-16)."""
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model


class Stage3GPT2Predictor(nn.Module):
    def __init__(self, d_model, codebook_size=64, num_rq_layers=3, num_classes=2,
                 n_heads=4, n_layers=2, dropout=0.1, use_pretrained=True,
                 pretrained_model='gpt2', freeze_gpt2=False, unfreeze_layers=12,
                 n_vars=28, bypass_gpt2_cls=False):
        super().__init__()
        if bypass_gpt2_cls:
            raise ValueError('Bypassing GPT-2 is incompatible with paper Eq. 15.')
        self.d_model, self.n_vars = d_model, n_vars
        self.num_rq_layers, self.codebook_size = num_rq_layers, codebook_size
        if use_pretrained:
            self.gpt2 = GPT2Model.from_pretrained(pretrained_model,
                resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
        else:
            self.gpt2 = GPT2Model(GPT2Config(vocab_size=codebook_size, n_embd=d_model,
                n_head=n_heads, n_layer=n_layers, n_positions=1024,
                resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))
        hidden = self.gpt2.config.n_embd
        # GPT2Model adds positional embeddings internally.
        self.input_proj = nn.Linear(n_vars, hidden)
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_rq_layers)])
        self.code_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden + level * d_model, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, codebook_size))
            for level in range(num_rq_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden + num_rq_layers * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
            for block in self.gpt2.h[max(0, len(self.gpt2.h) - unfreeze_layers):]:
                for p in block.parameters():
                    p.requires_grad_(True)
            for p in self.gpt2.ln_f.parameters():
                p.requires_grad_(True)

    def forward(self, x, codes=None):
        if x.ndim != 3 or x.shape[1] != self.n_vars:
            raise ValueError(f'Expected (batch, {self.n_vars}, time), got {tuple(x.shape)}')
        # Target codes are ignored; predictions depend only on x.
        context = self.gpt2(inputs_embeds=self.input_proj(x.transpose(1, 2))).last_hidden_state[:, -1]
        embeds, predicted, logits = [], [], []
        for level, head in enumerate(self.code_heads):
            scores = head(torch.cat([context] + embeds, dim=-1))
            # Hard argmax is the discrete feedback choice used here.
            code = scores.argmax(dim=-1)
            logits.append(scores)
            predicted.append(code)
            embeds.append(self.code_embeddings[level](code))
        return {'layer_logits': logits, 'stacked_logits': torch.stack(logits, 1),
                'predicted_codes': predicted, 'global_repr': context,
                'cls_logits': self.classifier(torch.cat([context] + embeds, -1))}

    def get_loss(self, outputs, codes, labels, cls_weight=1.0, code_weights=None):
        weights = [1.0] * len(codes) if code_weights is None else code_weights
        if len(codes) != self.num_rq_layers or len(weights) != len(codes):
            raise ValueError('Each RQ level requires one target and one weight.')
        cls_loss = F.cross_entropy(outputs['cls_logits'], labels)
        # Sum code losses over RQ levels (Eq. 16).
        pred_loss = sum(w * F.cross_entropy(logit, target)
            for w, logit, target in zip(weights, outputs['layer_logits'], codes))
        total = cls_weight * cls_loss + pred_loss
        return total, {'cls_loss': cls_loss.item(), 'pred_loss': pred_loss.item(), 'total_loss': total.item()}

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        gpt2 = sum(p.numel() for p in self.gpt2.parameters())
        return {'total': total, 'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'gpt2': gpt2, 'other': total - gpt2}
