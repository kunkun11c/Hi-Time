# FingerMovements

[configs/fingermovements.json](../configs/fingermovements.json) records K=64, L=3 and λ=20. These are reproducibility settings; the paper's exact historical configuration is unavailable. Scores are also provided in [fingermovements.csv](../results/fingermovements.csv).

| Seed | Correct / test examples | Accuracy |
| --- | --- | --- |
| 957 | 56 / 84 | 66.6667% |
| 1291 | 54 / 84 | 64.2857% |
| 1491 | 54 / 84 | 64.2857% |
| 1903 | 53 / 84 | 63.0952% |
| 1498 | 52 / 84 | 61.9048% |
| Mean | 269 / 420 predictions | **64.0476%** |

The population standard deviation is 1.5793 percentage points. Test subsets may overlap across seeds.

The 416 examples each have 28 variables and 50 time points. Official train/test arrays are merged and stratified into 249 training, 83 validation and 84 test examples. For each variable, normalization uses the training-set mean and population standard deviation across samples and time: `(x - mean) / (std + 1e-8)`. The run seed controls splitting and training.

The encoder uses width 128, 2 layers, 4 heads, FFN 256, patch length 8, stride 2, three FFT scales and dropout 0.1. Stage 1 trains for 150 epochs with AdamW, learning rate 1e-3, weight decay 1e-4, cosine scheduling, mixup 0.4 and label smoothing 0.1. Residual K-Means uses ten initializations and at most 50 Lloyd iterations per level.

Stage 3 fine-tunes all GPT-2 small layers, with dropout 0.1, AdamW learning rate 1e-5, weight decay 1e-4, at most 200 epochs, patience 80 and no learning-rate schedule. Batch size is 32. The loss is the sum of three code cross-entropies plus 20 times class cross-entropy. Checkpoints are selected by validation accuracy.

The recorded environment was Python 3.9.24, PyTorch 2.0.0+cu118, cuDNN 8.7 and Transformers 4.30.2 on NVIDIA L20Y. Both checkpoint replay and fresh training reproduced the recorded test probabilities in this environment; identical predictions on other hardware are not guaranteed.

From the repository root:

```bash
bash scripts/setup_paper_env.sh
HITIME_GPUS="0 1 2 3 4" bash scripts/reproduce_fingermovements.sh experiments/results/fm_five_seeds
```

Use `HITIME_GPUS="0"` for one GPU. See the [README](../README.md) for single-seed and replay commands and the result provenance.
