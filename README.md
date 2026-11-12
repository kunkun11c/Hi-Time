# Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification

Hi-Time learns temporal representations, quantizes them into coarse-to-fine semantic codes, and trains GPT-2 to predict those codes before classifying a time series. This version corrects the input and prediction paths in the earlier release. See [implementation details](docs/implementation.md).

![Hi-Time architecture](model.png)

## Installation

Use Python 3.9 and PyTorch 2.0.0 with CUDA 11.8. The setup script requires [uv](https://docs.astral.sh/uv/) and creates `.venv-torch20` without replacing the system Python.

```bash
bash scripts/setup_paper_env.sh
bash scripts/python_paper.sh -m unittest discover -s tests
```

Dependencies are pinned in [requirements-paper-lock.txt](requirements-paper-lock.txt). `scripts/python_paper.sh` runs the environment's Python and selects its PyTorch shared libraries. Set `HITIME_PAPER_ENV` if the environment is stored elsewhere.

Download the pretrained [GPT-2 small weights](https://huggingface.co/openai-community/gpt2) separately:

```text
pretrained/gpt2/
├── config.json
└── model.safetensors
```

## FingerMovements

The repository includes the [UEA FingerMovements dataset](https://www.timeseriesclassification.com/dataset.php). The recorded configuration is [configs/fingermovements.json](configs/fingermovements.json): K=64, L=3, λ=20, pretrained GPT-2, and GPT-2 learning rate 1e-5.

Run one seed:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_paper.sh experiments/run_finger.py \
  --config configs/fingermovements.json --seed 957 \
  --output experiments/results/fm_seed957 --paper-protocol
```

Run the five recorded seeds in parallel, one job per GPU:

```bash
HITIME_GPUS="0 1 2 3 4" bash scripts/reproduce_selected.sh \
  experiments/results/fm_five_seeds
```

Use `HITIME_GPUS="0"` to queue them on one GPU. Completed runs are retained when restarting the same command. The script saves checkpoints, data splits, normalization statistics, training histories, predictions, and environment records.

Replay a saved predictor using raw test sequences:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_paper.sh experiments/evaluate_checkpoint.py \
  --run-dir experiments/results/fm_seed957
```

The five recorded seeds average **64.0476%**. Both the configuration and seeds were selected using test accuracy; this is an exploratory result. Across all 27 tested seeds for that configuration, the mean is **55.0705%**. The [experiment record](docs/fingermovements.md) includes individual scores, the search scope, and replay checks. The original paper's exact historical configuration is unavailable.

## References

- [Original Hi-Time repository](https://github.com/qianlima-lab/Hi-Time)
- [TS-GAC benchmark](https://github.com/Frank-Wang-oss/TS-GAC)
