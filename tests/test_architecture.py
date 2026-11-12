"""Behavioral regression checks for the architecture described in Eqs. 13--16."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs import Config
from models.predictor import Stage3GPT2Predictor
from trainers.pipeline import ThreeStagePipeline


class FixedCodeHead(nn.Module):
    def __init__(self, code: int, vocabulary: int):
        super().__init__()
        self.code, self.vocabulary = code, vocabulary

    def forward(self, features):
        scores = features.new_full((features.size(0), self.vocabulary), -10.0)
        scores[:, self.code] = 10.0
        return scores


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, levels=3):
        torch.manual_seed(7)
        return Stage3GPT2Predictor(d_model=8, codebook_size=4,
                                  num_rq_layers=levels, num_classes=2,
                                  n_heads=2, n_layers=1, dropout=0.0,
                                  use_pretrained=False, n_vars=3).eval()

    def test_target_codes_cannot_change_predictions(self):
        model = self.make_model()
        x = torch.randn(2, 3, 9)
        targets_a = [torch.zeros(2, dtype=torch.long) for _ in range(3)]
        targets_b = [torch.full((2,), 3, dtype=torch.long) for _ in range(3)]
        with torch.no_grad():
            out_a = model(x, codes=targets_a)
            out_b = model(x, codes=targets_b)
            torch.testing.assert_close(out_a["stacked_logits"], out_b["stacked_logits"], rtol=0, atol=0)
            torch.testing.assert_close(out_a["cls_logits"], out_b["cls_logits"], rtol=0, atol=0)

    def test_raw_time_series_controls_predictions(self):
        model = self.make_model()
        x = torch.randn(2, 3, 9)
        with torch.no_grad():
            first = model(x)
            second = model(x + torch.tensor([0.0, 2.0, -3.0]).view(1, 3, 1))
        self.assertFalse(torch.allclose(first["global_repr"], second["global_repr"]))
        self.assertFalse(torch.allclose(first["cls_logits"], second["cls_logits"]))

    def test_code_prediction_depends_on_earlier_argmax_only(self):
        model = self.make_model()
        first_head = FixedCodeHead(0, 4)
        model.code_heads[0] = first_head
        # Make level 1's output a deterministic function of predicted level 0.
        following_head = nn.Linear(16, 4, bias=False)
        with torch.no_grad():
            model.code_embeddings[0].weight[0].fill_(-1)
            model.code_embeddings[0].weight[1].fill_(1)
            following_head.weight.zero_()
            following_head.weight[0, -8:] = 1
            following_head.weight[1, -8:] = -1
        model.code_heads[1] = following_head
        x = torch.randn(2, 3, 9)
        with torch.no_grad():
            out_a = model(x)
            first_head.code = 1
            out_b = model(x)
            self.assertTrue(torch.all(out_a["predicted_codes"][1] == 1))
            self.assertTrue(torch.all(out_b["predicted_codes"][1] == 0))
            # Editing a later predictor cannot influence earlier predictions.
            model.code_heads[2] = FixedCodeHead(3, 4)
            out_c = model(x)
            torch.testing.assert_close(out_b["layer_logits"][0], out_c["layer_logits"][0], rtol=0, atol=0)
            torch.testing.assert_close(out_b["layer_logits"][1], out_c["layer_logits"][1], rtol=0, atol=0)

    def test_four_level_loss_is_weighted_sum(self):
        model = self.make_model(levels=4)
        outputs = model(torch.randn(2, 3, 9))
        codes = [torch.tensor([level % 4, (level + 1) % 4]) for level in range(4)]
        labels = torch.tensor([0, 1])
        weights = [0.3, 0.5, 1.1, 0.8]
        total, metrics = model.get_loss(outputs, codes, labels, cls_weight=2.5, code_weights=weights)
        expected_code = sum(weight * F.cross_entropy(logits, target)
                            for weight, logits, target in zip(weights, outputs["layer_logits"], codes))
        expected = 2.5 * F.cross_entropy(outputs["cls_logits"], labels) + expected_code
        torch.testing.assert_close(total, expected)
        self.assertAlmostEqual(metrics["pred_loss"], expected_code.item(), places=6)
        total.backward()
        self.assertIsNotNone(model.input_proj.weight.grad)
        self.assertIsNotNone(model.code_heads[3][-1].weight.grad)

    def test_pipeline_evaluation_requires_only_predictor(self):
        model = self.make_model()
        data = torch.randn(4, 3, 9)
        labels = torch.tensor([0, 1, 0, 1])
        pipeline = ThreeStagePipeline(Config(), device=torch.device("cpu"))
        pipeline.stage1_trainer = None
        pipeline.discretizer = None
        pipeline.stage3_trainer = SimpleNamespace(model=model)
        with torch.no_grad():
            expected = (model(data)["cls_logits"].argmax(-1) == labels).float().mean().item()
        actual = pipeline.evaluate(DataLoader(TensorDataset(data, labels), batch_size=2))
        self.assertEqual(actual["accuracy"], expected)


if __name__ == "__main__":
    unittest.main()
