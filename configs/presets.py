from .base_config import Config, DataConfig, ModelConfig, TrainingConfig


def get_default_config() -> Config:
    """Reproduction starting point; the paper does not disclose every setting."""
    return Config(
        model=ModelConfig(
            d_model=128,
            codebook_size=64,
            num_quantize_layers=3,
            gpt2_model='./pretrained/gpt2',
        ),
    )


def get_fast_debug_config() -> Config:
    return Config(
        data=DataConfig(batch_size=4),
        model=ModelConfig(d_model=64, n_heads=2, n_layers=1, codebook_size=16, num_quantize_layers=1),
        training=TrainingConfig(
            stage1_epochs=5, stage2_epochs=3, stage3_epochs=5,
            stage1_patience=3, stage3_patience=3,
            num_runs=1, merge_resplit=False,
        ),
    )


def _paper_config(codebook_size: int, num_quantize_layers: int, batch_size: int = 32) -> Config:
    """Reproduction config with given K and L, not a recovered historical config."""
    return Config(
        data=DataConfig(batch_size=batch_size),
        model=ModelConfig(
            d_model=128,
            codebook_size=codebook_size,
            num_quantize_layers=num_quantize_layers,
            gpt2_model='./pretrained/gpt2',
        ),
    )


# Starting values: (K, L, batch_size), not recovered paper settings.
DATASET_PRESETS = {
    'ArticularyWordRecognition': (64, 3, 32),
    'FingerMovements':           (64, 3, 32),
    'SpokenArabicDigits':        (128, 3, 64),
    'CharacterTrajectories':     (64, 3, 32),
    'FaceDetection':             (64, 2, 64),
    'InsectWingbeat':            (128, 3, 64),
    'MotorImagery':              (64, 3, 16),
    'SelfRegulationSCP1':        (64, 3, 32),
}


PRESETS = {
    'default': get_default_config,
    'debug': get_fast_debug_config,
}

for _ds, (_k, _l, _bs) in DATASET_PRESETS.items():
    PRESETS[_ds.lower()] = lambda k=_k, l=_l, bs=_bs: _paper_config(k, l, bs)


def get_preset(name: str) -> Config:
    key = name if name in PRESETS else name.lower()
    if key not in PRESETS:
        available = ', '.join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]()
