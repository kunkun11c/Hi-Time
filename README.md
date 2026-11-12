# [ICML 2026] Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification

## Overview

This repository contains the implementation of *Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification*.

Hi-Time constructs coarse-to-fine prediction trajectories using temporal semantic codes. It first pre-trains a temporal encoder with multi-view fusion across scales, variables, and temporal patches. Residual K-Means then discretizes the learned representations into hierarchical semantic codes. Finally, GPT-2 takes projected time-series inputs and predicts these codes step by step before inferring the class label.

## The Model Framework of Hi-Time

![Hi-Time architecture](model.png)

## Datasets

This release includes FingerMovements from the [UEA multivariate time series archive](https://www.timeseriesclassification.com/dataset.php). The data is provided in the following format:

```text
data/datasets/FingerMovements/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

## Requirements

Use Python 3.9. The dependencies specify PyTorch 2.0.0 with CUDA 11.8 and Transformers 4.30.2.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pretrained GPT-2 Weights

Stage 3 requires pretrained GPT-2 small weights. Download them from [Hugging Face](https://huggingface.co/openai-community/gpt2) and place the files under `pretrained/gpt2/`:

```text
pretrained/gpt2/
├── config.json
└── model.safetensors
```

## Running the Code

All commands use the FingerMovements configuration in [configs/fingermovements.json](configs/fingermovements.json). Provide integer seed values through `--seed` or `--seeds`; replace the placeholders in the examples below.

### Training

Run the full three-stage pipeline with one seed:

```bash
python train.py --seed SEED --gpus 0
```

Run multiple seeds sequentially on one GPU:

```bash
python train.py --seeds SEED_1 SEED_2 SEED_3 SEED_4 SEED_5 --gpus 0
```

Run multiple seeds in parallel, with one task per GPU:

```bash
python train.py --seeds SEED_1 SEED_2 SEED_3 SEED_4 SEED_5 --gpus 0 1 2 3 4
```

### Available Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--config` | `configs/fingermovements.json` | Configuration file |
| `--seed` | Required unless `--seeds` is supplied | Seed for a single run |
| `--seeds` | Required unless `--seed` is supplied | Seeds for multiple runs |
| `--gpus` | `0` | GPU IDs; one concurrent task per GPU |
| `--output` | `outputs` | Output directory |
| `--threads` | `4` | CPU threads per task |

`--seed` and `--seeds` are mutually exclusive.
