from contextlib import contextmanager
import inspect
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base_trainer import BaseTrainer
from models.predictor import Stage3GPT2Predictor
from utils.data.datasets import TimeSeriesDataset


@contextmanager
def _preserve_random_states():
    """Preserve model, shuffle, and worker RNG streams during target caching."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def cache_training_codes(train_loader, encoder, discretizer, device):
    """Cache frozen targets for untransformed built-in tensor datasets.

    Keep the original sampler/generator and RNG states. Custom transforms or
    collation retain online target computation.
    """
    dataset = train_loader.dataset
    if train_loader.collate_fn is not default_collate:
        return train_loader
    if type(dataset) is TimeSeriesDataset and dataset.transform is None:
        inputs, labels = dataset.data, dataset.labels
    elif type(dataset) is TensorDataset and len(dataset.tensors) == 2:
        inputs, labels = dataset.tensors
    else:
        return train_loader
    if not len(inputs):
        return train_loader
    batch_size = (train_loader.batch_size
                  or getattr(train_loader.batch_sampler, 'batch_size', None)
                  or 32)
    modes = {module: module.training
             for root in (encoder, discretizer) for module in root.modules()}
    chunks = []
    try:
        with _preserve_random_states(), torch.no_grad():
            encoder.eval()
            discretizer.eval()
            for start in range(0, len(inputs), batch_size):
                x = inputs[start:start + batch_size].to(device)
                codes = discretizer(encoder.encode(x))['codes']
                if not chunks:
                    chunks = [[] for _ in codes]
                if len(codes) != len(chunks):
                    raise ValueError('Expert returned an inconsistent number of code levels')
                for level, target in enumerate(codes):
                    chunks[level].append(target.detach().cpu())
    finally:
        # Restore each expert submodule's original training mode.
        for module, training in modes.items():
            module.training = training
    cached = TensorDataset(inputs, labels, *(torch.cat(parts) for parts in chunks))
    # Reuse the sampler/generator to preserve indices and worker seeds.
    # Filter optional DataLoader arguments for older PyTorch versions.
    options = {'batch_sampler': train_loader.batch_sampler,
               'num_workers': train_loader.num_workers,
               'collate_fn': train_loader.collate_fn,
               'pin_memory': train_loader.pin_memory,
               'timeout': train_loader.timeout,
               'worker_init_fn': train_loader.worker_init_fn,
               'multiprocessing_context': train_loader.multiprocessing_context,
               'generator': train_loader.generator,
               'persistent_workers': train_loader.persistent_workers}
    if train_loader.num_workers:
        options['prefetch_factor'] = train_loader.prefetch_factor
    for name in ('pin_memory_device', 'in_order'):
        if hasattr(train_loader, name) and name in inspect.signature(DataLoader).parameters:
            options[name] = getattr(train_loader, name)
    return DataLoader(cached, **options)


class Stage3Trainer(BaseTrainer):
    def __init__(self, config, encoder, discretizer, device):
        super().__init__(config, device, patience=config.training.stage3_patience)
        self.encoder, self.discretizer = encoder.eval(), discretizer.eval()
        if config.training.stage3_mixup_alpha:
            raise ValueError('Legacy code-embedding mixup is not part of the paper architecture.')
        self.model = Stage3GPT2Predictor(
            d_model=config.model.d_model, codebook_size=config.model.codebook_size,
            num_rq_layers=config.model.num_quantize_layers, num_classes=config.data.num_classes,
            n_heads=config.model.n_heads, n_layers=config.model.n_layers,
            dropout=config.model.stage3_dropout, use_pretrained=config.model.use_pretrained_gpt2,
            pretrained_model=config.model.gpt2_model, freeze_gpt2=config.model.freeze_gpt2,
            unfreeze_layers=config.model.unfreeze_layers, n_vars=config.data.n_vars,
            bypass_gpt2_cls=config.model.bypass_gpt2_cls).to(device)
        print('[Stage 3] Paper hierarchical predictor:', self.model.get_num_parameters())
        self.cls_weight = config.training.loss_weight_label
        configured = [config.training.loss_weight_code1, config.training.loss_weight_code2,
                      config.training.loss_weight_code3]
        self.code_weights = [configured[min(i, 2)] for i in range(config.model.num_quantize_layers)]
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=config.training.stage3_lr, weight_decay=config.training.stage3_weight_decay)
        if config.training.stage3_use_scheduler:
            self.scheduler = CosineAnnealingLR(self.optimizer, config.training.stage3_epochs,
                eta_min=config.training.scheduler_eta_min)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = total_correct = total_samples = 0
        for batch in train_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            if len(batch) > 2:
                codes = [target.to(self.device) for target in batch[2:]]
            else:
                with torch.no_grad():
                    codes = self.discretizer(self.encoder.encode(x))['codes']
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(x)
            loss, _ = self.model.get_loss(outputs, codes, y, self.cls_weight, self.code_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (outputs['cls_logits'].argmax(-1) == y).sum().item()
            total_samples += len(y)
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = total_correct = total_samples = 0
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)['cls_logits']
            total_loss += F.cross_entropy(logits, y).item() * len(y)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_samples += len(y)
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}


def train_stage3(config, encoder, discretizer, train_loader, val_loader=None, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = cache_training_codes(train_loader, encoder, discretizer, device)
    trainer = Stage3Trainer(config, encoder, discretizer, device)
    trainer.train(train_loader, val_loader, config.training.stage3_epochs)
    return trainer
