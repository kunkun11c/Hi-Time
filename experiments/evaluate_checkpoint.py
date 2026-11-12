#!/usr/bin/env python3
"""Evaluate a saved predictor on its recorded test split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from configs import Config
from experiments.run_finger import apply_overrides, digest, predict, prepare_data, write_json
from models.predictor import Stage3GPT2Predictor
from utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Default: RUN/checkpoints/predictor.pt")
    parser.add_argument("--output", type=Path, help="Default: RUN/replay.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--probability-atol", type=float, default=1e-5)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    checkpoint_path = (args.checkpoint or run / "checkpoints/predictor.pt").resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"A saved Stage 3 checkpoint is required: {checkpoint_path}")
    config = Config()
    apply_overrides(config, json.loads((run / "config.json").read_text()))
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    set_seed(config.training.seed)
    with TemporaryDirectory(prefix="hitime-replay-") as temporary:
        temporary = Path(temporary)
        _, _, test_loader, test_indices = prepare_data(config, temporary)
        original_split = np.load(run / "split_and_normalization.npz")
        replay_split = np.load(temporary / "split_and_normalization.npz")
        for name in original_split.files:
            np.testing.assert_array_equal(original_split[name], replay_split[name],
                                          err_msg=f"Replayed data preparation differs: {name}")
    model = Stage3GPT2Predictor(
        d_model=config.model.d_model, codebook_size=config.model.codebook_size,
        num_rq_layers=config.model.num_quantize_layers, num_classes=config.data.num_classes,
        n_heads=config.model.n_heads, n_layers=config.model.n_layers,
        dropout=config.model.stage3_dropout, use_pretrained=config.model.use_pretrained_gpt2,
        pretrained_model=config.model.gpt2_model, freeze_gpt2=config.model.freeze_gpt2,
        unfreeze_layers=config.model.unfreeze_layers, n_vars=config.data.n_vars,
        bypass_gpt2_cls=config.model.bypass_gpt2_cls)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("config") != asdict(config):
        raise ValueError("Checkpoint configuration differs from run/config.json")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    labels, probabilities, predictions, accuracy = predict(model, test_loader, device)
    original = np.load(run / "test_predictions.npz")
    indices_match = bool(np.array_equal(test_indices, original["test_indices"]))
    labels_match = bool(np.array_equal(labels, original["labels"]))
    predictions_match = bool(np.array_equal(predictions, original["predictions"]))
    maximum_probability_difference = float(np.max(np.abs(probabilities - original["probabilities"])))
    probability_match = maximum_probability_difference <= args.probability_atol
    result = {
        "replayed_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path), "device": str(device),
        "torch": torch.__version__, "seed": config.training.seed,
        "test_accuracy": accuracy, "test_correct": int((predictions == labels).sum()),
        "test_total": len(labels), "split_and_normalization_exact": True,
        "test_indices_match": indices_match, "labels_match": labels_match,
        "predictions_match": predictions_match,
        "maximum_probability_difference": maximum_probability_difference,
        "probability_atol": args.probability_atol, "probabilities_match": probability_match,
        "expert_encoder_loaded": False, "discretizer_loaded": False,
        "passed": indices_match and labels_match and predictions_match and probability_match,
    }
    destination = (args.output or run / "replay.json").resolve()
    write_json(destination, result)
    np.savez_compressed(destination.with_name(destination.stem + "_predictions.npz"),
                        labels=labels, probabilities=probabilities,
                        predictions=predictions, test_indices=test_indices)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
