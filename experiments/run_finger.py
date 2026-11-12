#!/usr/bin/env python3
"""Train and record one complete FingerMovements run."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from configs import Config
from trainers.stage1_trainer import train_stage1
from trainers.stage2_trainer import train_stage2
from trainers.stage3_trainer import train_stage3
from utils.data.datasets import TimeSeriesDataset
from utils.data.loaders import load_uea_dataset
from utils.helpers import set_seed


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return str(error)


def apply_overrides(target, updates: dict, prefix: str = "") -> None:
    if not isinstance(updates, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    allowed = {field.name for field in fields(target)}
    for key, value in updates.items():
        if key not in allowed:
            raise ValueError(f"Unknown configuration field: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            apply_overrides(current, value, prefix + key + ".")
        else:
            setattr(target, key, value)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def stage_log(path: Path):
    with path.open("w", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def prepare_data(config: Config, output: Path):
    if not config.training.merge_resplit:
        raise ValueError("This runner implements only the merged, stratified 60/20/20 protocol")
    train_x, train_y, test_x, test_y = load_uea_dataset("FingerMovements", config.data.data_dir)
    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    if x.ndim != 3 or x.shape[1:] != (28, 50):
        raise ValueError(f"Expected FingerMovements in (N,28,50) format, received {x.shape}")
    indices = np.arange(len(x))
    train_indices, temporary = train_test_split(
        indices, test_size=0.4, random_state=config.training.seed, stratify=y)
    val_indices, test_indices = train_test_split(
        temporary, test_size=0.5, random_state=config.training.seed, stratify=y[temporary])
    mean = x[train_indices].mean(axis=(0, 2), keepdims=True)
    std = x[train_indices].std(axis=(0, 2), keepdims=True) + 1e-8
    np.savez_compressed(output / "split_and_normalization.npz",
                        train_indices=train_indices, val_indices=val_indices,
                        test_indices=test_indices, train_mean=mean, train_std=std,
                        normalization_applied=np.asarray(config.data.normalize))
    loaders = []
    for split_indices, shuffle in [(train_indices, True), (val_indices, False), (test_indices, False)]:
        split_x = x[split_indices]
        if config.data.normalize:
            split_x = (split_x - mean) / std
        loaders.append(DataLoader(TimeSeriesDataset(split_x, y[split_indices]),
                                  batch_size=config.data.batch_size, shuffle=shuffle,
                                  num_workers=config.data.num_workers))
    config.data.seq_len = x.shape[2]
    config.data.n_vars = x.shape[1]
    config.data.num_classes = len(np.unique(y))
    return (*loaders, test_indices)


def environment_record(config: Config, device: torch.device, output: Path) -> dict:
    source_files = sorted(path for directory in ["configs", "models", "trainers", "utils", "experiments"]
                          for path in (ROOT / directory).rglob("*.py"))
    source_files += [ROOT / "train.py", ROOT / "requirements.txt", ROOT / "requirements-paper.txt", ROOT / "requirements-paper-lock.txt"]
    source_files += sorted((ROOT / "scripts").glob("*.sh"))
    source_files += sorted((ROOT / "scripts").glob("*.py"))
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in source_files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    (output / "source.patch").write_text(command_output(["git", "diff", "HEAD", "--"]))
    data_dir = Path(config.data.data_dir) / "FingerMovements"
    data_hashes = {path.name: digest(path) for path in sorted(data_dir.glob("*.npy"))}
    weights = Path(config.model.gpt2_model)
    weight_hashes = {}
    if weights.is_dir():
        weight_hashes = {path.name: digest(path) for path in weights.iterdir()
                         if path.suffix in {".json", ".safetensors", ".bin"}}
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "torch_module_file": torch.__file__,
        "platform": platform.platform(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions()
                                if dist.metadata["Name"])),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numerics": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "default_dtype": str(torch.get_default_dtype()),
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"]),
        "torch_threads": torch.get_num_threads(),
        "upstream_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_files},
        "data_sha256": data_hashes,
        "gpt2_files_sha256": weight_hashes,
        "checkpoint_selection": "highest validation accuracy; latest checkpoint on ties (repository EarlyStopping)",
        "test_selection_notice": "If configuration or seed is selected using this test accuracy, report it as test-selected exploratory evaluation.",
    }


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, stage1: bool = False):
    model.eval()
    all_labels, all_probabilities = [], []
    for x, y in loader:
        outputs = model(x.to(device))
        logits = outputs["logits" if stage1 else "cls_logits"]
        all_labels.append(y.numpy())
        all_probabilities.append(logits.softmax(dim=-1).float().cpu().numpy())
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=-1)
    return labels, probabilities, predictions, float(np.mean(predictions == labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON with nested Config overrides")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-checkpoints", action="store_true", help="Skip large model weight files during sweeps")
    parser.add_argument("--paper-protocol", action="store_true", help="Enforce disclosed paper settings and PyTorch 2.0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "config.json").exists():
        raise FileExistsError(f"Output already contains an experiment: {output}")
    start = time.monotonic()
    config = Config()
    if args.config:
        apply_overrides(config, json.loads(args.config.read_text()))
    if config.data.dataset_name != "FingerMovements":
        raise ValueError("This runner only supports FingerMovements")
    config.training.seed = args.seed
    config.training.num_runs = 1
    config.checkpoint_dir = str(output / "checkpoints")
    config.data.data_dir = str((ROOT / config.data.data_dir).resolve())
    if (ROOT / config.model.gpt2_model).is_dir():
        config.model.gpt2_model = str((ROOT / config.model.gpt2_model).resolve())
    if args.paper_protocol:
        from experiments.paper_constraints import check_paper_protocol
        write_json(output / "paper_protocol.json", check_paper_protocol(config, torch.__version__))
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, test_indices = prepare_data(config, output)
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", environment_record(config, device, output))
    state = {"status": "running", "seed": args.seed, "stages": {},
             "split_sizes": {"train": len(train_loader.dataset), "val": len(val_loader.dataset), "test": len(test_loader.dataset)}}
    write_json(output / "status.json", state)

    def checkpoint(name, model):
        if not args.no_checkpoints:
            directory = output / "checkpoints"
            directory.mkdir(exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config),
                        "seed": args.seed}, directory / f"{name}.pt")

    try:
        stage_start = time.monotonic()
        with stage_log(output / "stage1.log"):
            stage1 = train_stage1(config, train_loader, val_loader, device)
        write_json(output / "stage1_history.json", dict(stage1.metric_tracker.history))
        checkpoint("encoder", stage1.model)
        state["stages"]["stage1"] = {"seconds": time.monotonic() - stage_start,
                                             "best_val_accuracy": stage1.early_stopping.best_score}
        write_json(output / "status.json", state)

        set_seed(args.seed)
        stage_start = time.monotonic()
        with stage_log(output / "stage2.log"):
            discretizer = train_stage2(config, stage1.model, train_loader, val_loader, device)
        checkpoint("discretizer", discretizer)
        state["stages"]["stage2"] = {"seconds": time.monotonic() - stage_start}
        write_json(output / "status.json", state)

        set_seed(args.seed)
        stage_start = time.monotonic()
        with stage_log(output / "stage3.log"):
            stage3 = train_stage3(config, stage1.model, discretizer, train_loader, val_loader, device)
        write_json(output / "stage3_history.json", dict(stage3.metric_tracker.history))
        checkpoint("predictor", stage3.model)
        state["stages"]["stage3"] = {"seconds": time.monotonic() - stage_start,
                                             "best_val_accuracy": stage3.early_stopping.best_score}
        labels, probabilities, predictions, test_accuracy = predict(stage3.model, test_loader, device)
        _, _, _, val_accuracy = predict(stage3.model, val_loader, device)
        _, stage1_probabilities, stage1_predictions, stage1_accuracy = predict(stage1.model, test_loader, device, stage1=True)
        np.savez_compressed(output / "test_predictions.npz", labels=labels, probabilities=probabilities,
                            predictions=predictions, test_indices=test_indices,
                            stage1_probabilities=stage1_probabilities, stage1_predictions=stage1_predictions)
        result = {**state, "status": "complete", "test_accuracy": test_accuracy,
                  "test_correct": int((predictions == labels).sum()), "test_total": len(labels),
                  "validation_accuracy": val_accuracy, "stage1_test_accuracy": stage1_accuracy,
                  "elapsed_seconds": time.monotonic() - start,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "checkpoints_saved": not args.no_checkpoints}
        write_json(output / "result.json", result)
        write_json(output / "status.json", result)
        print(json.dumps(result, indent=2))
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        state.update(status="failed", elapsed_seconds=time.monotonic() - start)
        write_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
