import unittest
from configs import Config
from experiments.paper_constraints import check_paper_protocol


class PaperProtocolTests(unittest.TestCase):
    def test_disclosed_values_rejected_if_changed(self):
        config = Config()
        self.assertTrue(check_paper_protocol(config, '2.0.0+cu118')['validated'])
        for section, name, changed in [
            ('training', 'stage3_lr', 5e-5),
            ('model', 'codebook_size', 48),
            ('model', 'num_quantize_layers', 4),
            ('training', 'loss_weight_code1', 0.1),
        ]:
            target = getattr(config, section)
            original = getattr(target, name)
            setattr(target, name, changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_paper_protocol(config, '2.0.0+cu118')
            setattr(target, name, original)
        with self.assertRaises(ValueError):
            check_paper_protocol(config, '2.9.0')
