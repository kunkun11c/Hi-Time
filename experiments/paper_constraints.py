"""Validate the recorded experimental settings."""
import json
from pathlib import Path


def check_paper_protocol(config, torch_version):
    errors = []
    def require(condition, description):
        if not condition:
            errors.append(description)
    require(torch_version.split('+')[0] == '2.0.0', 'PyTorch must be 2.0.0 in this reproduction')
    require(config.training.stage3_lr == 1e-5, 'GPT-2 learning rate must be 1e-5')
    require(config.training.stage1_lr == 1e-3, 'Non-PLM encoder learning rate held at 1e-3')
    require(config.training.stage1_weight_decay == 1e-4, 'Encoder weight decay held at 1e-4')
    require(config.training.stage3_weight_decay == 1e-4, 'GPT-2 weight decay held at repository value 1e-4')
    require(config.training.stage1_use_scheduler, 'Non-PLM encoder uses cosine annealing')
    require(config.model.codebook_size in [16, 32, 64, 128, 256], 'K must be in the paper search grid')
    require(config.model.num_quantize_layers in [1, 2, 3], 'L must be in the paper search grid')
    require(config.model.use_pretrained_gpt2, 'Use pretrained GPT-2')
    require(not config.model.bypass_gpt2_cls, 'Classification must use GPT-2 features')
    require(config.training.loss_weight_code1 == config.training.loss_weight_code2 == config.training.loss_weight_code3 == 1.0,
            'Eq.16 code CE coefficients must be one; only lambda may be tuned')
    require(config.training.loss_weight_label > 0, 'Classification lambda must be positive')
    require(config.training.merge_resplit, 'Use merged 60/20/20 reproduction split')
    path = Path(config.model.gpt2_model) / 'config.json'
    if path.is_file():
        gpt_config = json.loads(path.read_text())
        require(gpt_config.get('n_layer') == 12 and gpt_config.get('n_embd') == 768 and gpt_config.get('n_head') == 12,
                'Preserve default GPT-2 small architecture: 12 layers, width 768, 12 heads')
    else:
        require(False, 'Recorded local pretrained GPT-2 configuration is required')
    if errors:
        raise ValueError('Paper protocol violation: ' + '; '.join(errors))
    return {'validated': True, 'torch': torch_version, 'gpt2_lr': config.training.stage3_lr,
            'gpt2_layers': 12, 'gpt2_hidden': 768, 'gpt2_heads': 12,
            'encoder_lr': 1e-3, 'encoder_weight_decay': 1e-4,
            'encoder_scheduler': 'CosineAnnealingLR',
            'codebook_search_grid': [16, 32, 64, 128, 256], 'depth_search_grid': [1, 2, 3],
            'code_loss': 'sum of unit-weight cross entropies',
            'split_ratios': [0.6, 0.2, 0.2], 'final_aggregate_runs': 5,
            'implementation_choices_not_disclosed_by_paper': [
                'PyTorch patch release 2.0.0 and CUDA 11.8; Python 3.9 follows repository README',
                'GPT-2 weight decay retained at 1e-4; not independently specified for PLM in paper',
                'Merge official splits, stratification, per-run split RNG, train-only per-variable z-score',
                'FFT scaling/alignment, hard argmax code embeddings, MLP widths, batch and epoch counts'],
            'hardware_note': 'GPU model is recorded in environment.json',
            'selection_notice': 'Configuration/seed selection may use test accuracy report as exploratory.'}
