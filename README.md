# Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification

Hi-Time first learns temporal representations, then generates hierarchical semantic codes with residual K-Means, and finally trains GPT-2 to predict the codes and class label.

![Hi-Time architecture](model.png)

## Installation

Use Python 3.9. The requirements pin PyTorch 2.0.0 with CUDA 11.8 and Transformers 4.30.2.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download [GPT-2 small](https://huggingface.co/openai-community/gpt2) separately and place `config.json` and `model.safetensors` in `pretrained/gpt2/`.

## Training

The default configuration is [configs/fingermovements.json](configs/fingermovements.json): K=64, L=3, λ=20, batch size 32, and GPT-2 learning rate 1e-5.

Supply seeds explicitly with `--seed` or `--seeds`. Replace the placeholders below with integer values. Runs can be parallelized across GPUs or queued on one GPU:

```bash
python train.py --seeds SEED_1 SEED_2 SEED_3 SEED_4 SEED_5 --gpus 0 1 2 3 4
python train.py --seeds SEED_1 SEED_2 SEED_3 SEED_4 SEED_5 --gpus 0
```

Run a single seed or choose an output directory:

```bash
python train.py --seed SEED --gpus 0 --output outputs/single_seed
python train.py --config configs/fingermovements.json --seeds SEED_1 SEED_2 --gpus 0 --output outputs/fingermovements
```

By default, each run writes `outputs/seed_<seed>/results.json`, `train.log`, saved predictions and `checkpoints/{encoder,discretizer,predictor}.pt`. The supplied runs' combined scores and mean are written to `outputs/results.json`; `--output` changes this root directory.

The bundled FingerMovements data is merged and stratified into 249 training, 83 validation and 84 test examples. Per-variable normalization uses only training samples; each seed controls both splitting and training. Checkpoints are selected by validation accuracy.

FFT downsampling steps use `round(max(selected_f) / f)`, and shared-backbone patch features are aligned by adaptive average pooling. GPT-2 receives projected raw time points; subsequent MLPs use embeddings of their own hard-argmax code predictions. FFT scaling, patch alignment and discrete-code handling are implementation choices where the paper leaves details unspecified.

## References

- [Original Hi-Time repository](https://github.com/qianlima-lab/Hi-Time)
- [TS-GAC benchmark](https://github.com/Frank-Wang-oss/TS-GAC)
