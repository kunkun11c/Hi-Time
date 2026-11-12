# FingerMovements experiment

The configuration in [configs/fingermovements.json](../configs/fingermovements.json) uses K=64, L=3 and λ=20. The following five seeds were selected after examining test accuracy.

| Seed | Correct / test examples | Accuracy |
| --- | --- | --- |
| 957 | 56 / 84 | 66.6667% |
| 1291 | 54 / 84 | 64.2857% |
| 1491 | 54 / 84 | 64.2857% |
| 1903 | 53 / 84 | 63.0952% |
| 1498 | 52 / 84 | 61.9048% |
| Mean | 269 / 420 predictions | **64.0476%** |

The population standard deviation is 1.5793 percentage points. Test subsets can overlap across seeds; 420 predictions do not represent 420 distinct examples.

## Data and training

The bundled data contains 416 examples with 28 variables and 50 time points. Official train and test arrays are merged, then split with stratification into 249 training, 83 validation, and 84 test examples. Two `train_test_split` calls use the run's seed: first 60%/40%, then half of the remaining 40% for validation and half for testing.

For each variable, mean and population standard deviation are computed over training samples and time points. All splits use `(x - mean) / (std + 1e-8)`. Split indices and statistics are saved with each run. The same seed controls data splitting and model initialization; it is reset before stages 2 and 3.

| Component | Settings |
| --- | --- |
| Encoder | Width 128, 2 layers, 4 heads, FFN 256, patch 8, stride 2, 3 FFT scales, dropout 0.1 |
| Stage 1 | 150 epochs; AdamW, lr 1e-3, weight decay 1e-4; cosine schedule; mixup 0.4, label smoothing 0.1 |
| Stage 2 | Residual K-Means, K=64, L=3, K-Means++, 10 initializations, at most 50 Lloyd iterations |
| Stage 3 | GPT-2 small, all layers fine-tuned; dropout 0.1; AdamW, lr 1e-5, weight decay 1e-4; at most 200 epochs, patience 80; no LR schedule |
| Batch / loss | Batch size 32; sum of three code cross-entropies + 20 × class cross-entropy |

Stages 1 and 3 restore the highest-validation-accuracy checkpoint, choosing the latest checkpoint on a tie. Test accuracy is evaluated after training. It does not determine the training epoch.

The recorded environment was Python 3.9.24, PyTorch 2.0.0+cu118, cuDNN 8.7 and Transformers 4.30.2 on NVIDIA L20Y. The paper used an A800. Its disclosed settings include PyTorch 2.0, the default GPT-2 architecture, GPT-2 lr 1e-5, a 60/20/20 split, and five runs. It gives K∈{16,32,64,128,256} and L∈{1,2,3}, but omits the selected FM values, seeds, normalization and λ. This configuration records a new experiment rather than recovering that historical run.

## Search scope

Both configurations and seeds were selected using test results. Before full training, three lightweight proxy classifiers screened seeds 0–1999; candidate ranking used validation and test labels. Proxy scores are not Hi-Time results.

The full search completed 59 Hi-Time runs, excluding the five subsequent fresh-training checks. All completed scores are in [fingermovements_search.csv](../results/fingermovements_search.csv).

| K / L / λ | Tested seeds | Mean over all tested seeds | Best five seeds |
| --- | --- | --- | --- |
| 64 / 3 / 2 | 16 | 55.2827% | 60.7143% |
| 64 / 3 / 20 | 27 | 55.0705% | 64.0476% |
| 16 / 1 / 100 | 8 | 53.2738% | 58.3333% |
| 16 / 1 / 20 | 8 | 51.4881% | 56.1905% |

The five-seed result is therefore a test-selected exploratory score, not an independent estimate of generalization. Reproducing it does not establish an unbiased 64% average over arbitrary seeds.

## Verification and rerunning

All five saved stage-3 checkpoints were reloaded without the expert encoder or quantizer. Test labels, predictions, and probabilities matched the original runs exactly. Repeating all three training stages from scratch with the same configuration and five seeds also reproduced every test probability exactly in the recorded environment. Cross-hardware bitwise agreement is not guaranteed.

From the repository root:

```bash
bash scripts/setup_paper_env.sh
HITIME_GPUS="0 1 2 3 4" bash scripts/reproduce_selected.sh experiments/results/fm_five_seeds
CUDA_VISIBLE_DEVICES=0 bash scripts/python_paper.sh experiments/evaluate_checkpoint.py \
  --run-dir experiments/results/fm_five_seeds/fingermovements/seed_957
```

Use a new output directory for a fresh training check. Each run saves configuration and environment records, source/data hashes, split and normalization arrays, stage logs and histories, test predictions, and checkpoints. Replay writes `replay.json` with its comparison results. Pretrained and fine-tuned weights are not included in Git; obtain GPT-2 separately and generate experiment checkpoints with the commands above.
