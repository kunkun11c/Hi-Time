"""Caching frozen expert targets must preserve training behavior and RNG streams."""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs import Config
from trainers.stage3_trainer import Stage3Trainer, cache_training_codes
from utils.data.datasets import TimeSeriesDataset


class FakeEncoder(nn.Module):
    def __init__(self, consume_random=False):
        super().__init__()
        self.consume_random = consume_random
        self.calls = 0

    def encode(self, x):
        self.calls += 1
        if self.consume_random:
            torch.rand((), device=x.device)
            random.random()
            np.random.rand()
        return x.mean(-1)


class FakeDiscretizer(nn.Module):
    def forward(self, features):
        first = (features.abs().sum(-1) * 7).long() % 4
        second = (features[:, 0].abs() * 11).long() % 4
        return {'codes': [first, second]}


class TinyPredictor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = nn.Linear(12, 2)
        self.heads = nn.ModuleList([nn.Linear(12, 4) for _ in range(2)])

    def forward(self, x):
        flattened = x.flatten(1)
        return {'cls_logits': self.classifier(flattened),
                'layer_logits': [head(flattened) for head in self.heads]}

    def get_loss(self, output, codes, y, cls_weight, code_weights):
        loss = cls_weight * F.cross_entropy(output['cls_logits'], y)
        loss = loss + sum(weight * F.cross_entropy(scores, target)
            for weight, scores, target in zip(code_weights, output['layer_logits'], codes))
        return loss, {}

    def get_num_parameters(self):
        return {'total': sum(parameter.numel() for parameter in self.parameters())}


class Stage3CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(321)
        self.x = torch.randn(17, 3, 4)
        self.y = torch.arange(17) % 2
        self.encoder = FakeEncoder()
        self.discretizer = FakeDiscretizer()

    def loader(self, **kwargs):
        return DataLoader(TensorDataset(self.x, self.y), **kwargs)

    def test_target_codes_match_online_and_own_training_modes_restored(self):
        loader = self.loader(batch_size=5, shuffle=False)
        self.encoder.train()
        self.discretizer.train()
        cached = cache_training_codes(loader, self.encoder, self.discretizer, torch.device('cpu'))
        self.assertTrue(self.encoder.training)
        self.assertTrue(self.discretizer.training)
        self.assertIs(cached.dataset.tensors[0], self.x)
        self.assertIs(cached.dataset.tensors[1], self.y)
        for ordinary, extended in zip(loader, cached):
            expected = self.discretizer(self.encoder.encode(ordinary[0]))['codes']
            for target, code in zip(extended[2:], expected):
                torch.testing.assert_close(target, code, atol=0, rtol=0)

    def test_all_random_streams_and_loader_generator_are_unchanged(self):
        self.encoder.consume_random = True
        generator = torch.Generator().manual_seed(87)
        loader = self.loader(batch_size=4, shuffle=True, generator=generator)
        cpu_state, generator_state = torch.get_rng_state(), generator.get_state()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cache_training_codes(loader, self.encoder, self.discretizer, torch.device('cpu'))
        torch.testing.assert_close(torch.get_rng_state(), cpu_state, atol=0, rtol=0)
        torch.testing.assert_close(generator.get_state(), generator_state, atol=0, rtol=0)
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])

    def test_shuffle_sequence_is_identical_with_and_without_explicit_generator(self):
        for explicit_generator in (False, True):
            generator = torch.Generator().manual_seed(87) if explicit_generator else None
            loader = self.loader(batch_size=4, shuffle=True, drop_last=True, generator=generator)
            cached = cache_training_codes(loader, self.encoder, self.discretizer, torch.device('cpu'))
            self.assertIs(cached.batch_sampler, loader.batch_sampler)
            self.assertIs(cached.generator, loader.generator)
            cpu_state = torch.get_rng_state()
            generator_state = generator.get_state() if generator is not None else None
            expected = [batch[0].clone() for _ in range(2) for batch in loader]
            torch.set_rng_state(cpu_state)
            if generator is not None:
                generator.set_state(generator_state)
            actual = [batch[0] for _ in range(2) for batch in cached]
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_augmentation_and_custom_collation_keep_online_path(self):
        augmented = TimeSeriesDataset(self.x.numpy(), self.y.numpy(), transform=lambda x: x + torch.randn_like(x))
        augmented_loader = DataLoader(augmented, batch_size=4)
        self.assertIs(cache_training_codes(augmented_loader, self.encoder, self.discretizer, 'cpu'),
                      augmented_loader)
        custom_loader = self.loader(batch_size=4, collate_fn=lambda batch: batch)
        self.assertIs(cache_training_codes(custom_loader, self.encoder, self.discretizer, 'cpu'),
                      custom_loader)
        self.assertEqual(self.encoder.calls, 0)

    def test_one_update_loss_gradients_and_weights_match(self):
        config = Config()
        config.model.num_quantize_layers = 2
        config.training.stage3_use_scheduler = False
        config.training.stage3_mixup_alpha = 0
        ordinary = self.loader(batch_size=len(self.x), shuffle=False)
        cached = cache_training_codes(ordinary, self.encoder, self.discretizer, 'cpu')
        with patch('trainers.stage3_trainer.Stage3GPT2Predictor', TinyPredictor):
            first = Stage3Trainer(config, self.encoder, self.discretizer, torch.device('cpu'))
        second = copy.deepcopy(first)
        loss_online = first.train_epoch(ordinary)
        before_calls = second.encoder.calls
        loss_cached = second.train_epoch(cached)
        self.assertEqual(second.encoder.calls, before_calls)
        self.assertEqual(loss_online, loss_cached)
        for online, precomputed in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(online.grad, precomputed.grad, atol=0, rtol=0)
            torch.testing.assert_close(online, precomputed, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_random_stream_is_unchanged(self):
        device = torch.device('cuda:0')
        torch.rand((), device=device)  # Initialize CUDA before taking the snapshot.
        states = torch.cuda.get_rng_state_all()
        self.encoder.consume_random = True
        cache_training_codes(self.loader(batch_size=5), self.encoder,
                             self.discretizer, device)
        for before, after in zip(states, torch.cuda.get_rng_state_all()):
            torch.testing.assert_close(before, after, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
