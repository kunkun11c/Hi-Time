"""Architecture invariants for paper Eqs. 1-12, using only synthetic data."""
import unittest

import torch

from models.encoder import Stage1Encoder
from models.components.attention import CrossVariableFusion, GatedPatchFusion
from models.components.quantizers import ResidualQuantizer


class Stage12ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def encoder(self):
        return Stage1Encoder(50, 4, 2, [1, 2, 4], 8, 2, 16, 2, 1, 32, 0).eval()

    def test_shared_backbone_gradients_and_batch_invariance(self):
        model = self.encoder()
        x = torch.randn(4, 4, 50)
        together = model(x)['repr']
        separately = torch.cat([model(sample.unsqueeze(0))['repr'] for sample in x])
        self.assertEqual(together.shape, (4, 16))
        torch.testing.assert_close(together, separately, atol=2e-6, rtol=1e-5)
        together.square().mean().backward()
        self.assertGreater(model.multi_scale.backbone.patch_embed.proj.weight.grad.norm(), 0)
        for mlp in model.multi_scale.scale_mlps:
            self.assertGreater(mlp[0].weight.grad.norm(), 0)

    def test_fft_dominant_frequency_reciprocal_steps(self):
        t = torch.arange(50).float()
        signal = (5 * torch.sin(2 * torch.pi * 2 * t / 50)
                  + 3 * torch.sin(2 * torch.pi * 4 * t / 50)
                  + torch.sin(2 * torch.pi * 10 * t / 50)).view(1, 1, 50)
        self.assertEqual(self.encoder().multi_scale.dominant_steps(signal).tolist(), [[5, 2, 1]])

    def test_cross_variable_residual_identity(self):
        fusion = CrossVariableFusion(16, 2)
        with torch.no_grad():
            for parameter in fusion.parameters():
                parameter.zero_()
        x = torch.randn(3, 4, 7, 16)
        torch.testing.assert_close(fusion(x), x, atol=0, rtol=0)

    def test_gate_uniform_scores_equal_global_mean(self):
        fusion = GatedPatchFusion(16)
        with torch.no_grad():
            for parameter in fusion.gate.parameters():
                parameter.zero_()
        x = torch.randn(3, 4, 7, 16)
        torch.testing.assert_close(fusion(x), x.mean((1, 2)), atol=1e-6, rtol=1e-5)

    def test_residual_lloyd_reconstruction_and_frozen_codebooks(self):
        quantizer = ResidualQuantizer(16, 8, 3)
        features = torch.randn(80, 16)
        quantizer.fit(features, max_iter=25, seed=123)
        state = {name: value.clone() for name, value in quantizer.state_dict().items()}
        output = quantizer(features)
        errors = [features.square().mean()] + [r.square().mean() for r in output['residuals']]
        for previous, current in zip(errors, errors[1:]):
            self.assertLessEqual(current, previous + 1e-6)
        torch.testing.assert_close(quantizer.decode(output['codes']), output['quantized'])
        torch.testing.assert_close(features - output['quantized'], output['residuals'][-1],
                                   atol=1e-6, rtol=1e-5)
        for level, codebook in enumerate(quantizer.quantizers):
            torch.testing.assert_close(codebook.codebook, quantizer.code_embeddings[level].weight)
        quantizer.train()(torch.randn(4, 16))
        quantizer.eval()(torch.randn(4, 16))
        for name, value in quantizer.state_dict().items():
            torch.testing.assert_close(state[name], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
