# Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification

Hi-Time learns temporal representations, quantizes them into coarse-to-fine semantic codes, and trains GPT-2 to predict those codes before classifying a time series. See [implementation details](docs/implementation.md).

![Hi-Time architecture](model.png)

## Installation

The setup script requires [uv](https://docs.astral.sh/uv/) and creates a Python 3.9 environment with PyTorch 2.0.0 and CUDA 11.8. Dependencies are pinned in [requirements-paper-lock.txt](requirements-paper-lock.txt).

```bash
bash scripts/setup_paper_env.sh
```

Download [GPT-2 small](https://huggingface.co/openai-community/gpt2) separately and place `config.json` and `model.safetensors` in `pretrained/gpt2/`. Use `scripts/python_paper.sh` to run the installed environment; set `HITIME_PAPER_ENV` if it is stored elsewhere.

## FingerMovements

The bundled [UEA FingerMovements dataset](https://www.timeseriesclassification.com/dataset.php) uses [configs/fingermovements.json](configs/fingermovements.json): K=64, L=3, λ=20, and pretrained GPT-2 with learning rate 1e-5.

Run the five recorded seeds in parallel, one job per GPU:

```bash
HITIME_GPUS="0 1 2 3 4" bash scripts/reproduce_fingermovements.sh \
  experiments/results/fm_five_seeds
```

Use `HITIME_GPUS="0"` to run them sequentially on one GPU. Run a single seed with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_paper.sh experiments/run_finger.py \
  --config configs/fingermovements.json --seed 957 \
  --output experiments/results/fm_seed957 --paper-protocol
```

Each run saves checkpoints, data splits, normalization statistics, training histories, predictions, and environment records. Replay a saved predictor with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_paper.sh experiments/evaluate_checkpoint.py \
  --run-dir experiments/results/fm_seed957
```

The five-seed mean is **64.0476%**; individual scores and settings are in the [experiment notes](docs/fingermovements.md) and [results CSV](results/fingermovements.csv). The configuration and seeds were selected using test accuracy; these results document reproducibility rather than an independent evaluation.

## References

- [Original Hi-Time repository](https://github.com/qianlima-lab/Hi-Time)
- [TS-GAC benchmark](https://github.com/Frank-Wang-oss/TS-GAC)
