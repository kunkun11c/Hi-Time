#!/usr/bin/env python3
"""Run configurations and seeds across available GPUs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def aggregate(tasks: list[dict], output: Path):
    rows = []
    grouped = {}
    for task in tasks:
        result = read_json(task["directory"] / "result.json")
        status = read_json(task["directory"] / "status.json") or {}
        row = {"config": task["name"], "seed": task["seed"], "status": task["status"],
               "test_accuracy": None, "validation_accuracy": None,
               "stage1_test_accuracy": None, "elapsed_seconds": status.get("elapsed_seconds"),
               "output_directory": str(task["directory"])}
        if result and result.get("status") == "complete":
            for key in ["test_accuracy", "validation_accuracy", "stage1_test_accuracy", "elapsed_seconds"]:
                row[key] = result.get(key)
            row["status"] = "complete"
        rows.append(row)
        grouped.setdefault(task["name"], []).append(row)
    with (output / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    configs = {}
    for name, config_rows in grouped.items():
        finished = [row for row in config_rows if row["status"] == "complete"]
        accuracies = [row["test_accuracy"] for row in finished]
        configs[name] = {
            "requested_seeds": [row["seed"] for row in config_rows],
            "completed_count": len(finished), "requested_count": len(config_rows),
            "all_requested_seeds_complete": len(finished) == len(config_rows),
            "all_requested_seeds_mean": statistics.mean(accuracies) if len(finished) == len(config_rows) else None,
            "completed_seeds_mean": statistics.mean(accuracies) if accuracies else None,
            "completed_seeds_std_ddof0": statistics.pstdev(accuracies) if accuracies else None,
            "completed_results": [{"seed": row["seed"], "test_accuracy": row["test_accuracy"]} for row in finished],
        }
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {state: sum(row["status"] == state for row in rows)
                   for state in ["pending", "running", "complete", "failed", "interrupted"]},
        "evaluation_note": "Summaries include every completed run for the supplied configuration and seed list.",
        "checkpoint_selection": "Validation accuracy only, per run_finger.py; test accuracy is not used to choose epochs.",
        "configs": configs,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--paper-protocol", action="store_true", help="Enforce paper constraints in every worker")
    parser.add_argument("--interleave-configs", action="store_true", help="Alternate configurations across seeds")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1 or args.threads < 1 or args.poll_seconds <= 0:
        parser.error("jobs-per-gpu, threads, and poll-seconds must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.gpus)) != len(args.gpus):
        parser.error("seeds and GPU IDs must be unique")
    if len({path.stem for path in args.configs}) != len(args.configs):
        parser.error("configuration file stems must be unique")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for path in args.configs:
        path = path.resolve()
        contents = path.read_bytes()
        config = json.loads(contents)
        if config.get("data", {}).get("dataset_name", "FingerMovements") != "FingerMovements":
            parser.error(f"Only FingerMovements is supported: {path}")
        fingerprint = hashlib.sha256(contents).hexdigest()
        config_copy = output / "config_inputs" / f"{path.stem}_{fingerprint[:12]}.json"
        config_copy.parent.mkdir(exist_ok=True)
        if not config_copy.exists():
            config_copy.write_bytes(contents)
        for seed in args.seeds:
            directory = output / path.stem / f"seed_{seed}"
            manifest = read_json(directory / "launcher_task.json")
            if manifest and manifest.get("input_config_sha256") != fingerprint:
                raise ValueError(f"Configuration changed since this output was created: {directory}")
            result = read_json(directory / "result.json")
            completed = bool(result and result.get("status") == "complete")
            if completed and not manifest:
                raise ValueError(f"Completed output lacks launcher provenance: {directory}")
            tasks.append({"name": path.stem, "config": config_copy, "original_config": path, "seed": seed,
                          "fingerprint": fingerprint, "directory": directory,
                          "status": "complete" if completed else "pending"})
    if args.interleave_configs:
        seed_order = {seed: index for index, seed in enumerate(args.seeds)}
        config_order = {path.stem: index for index, path in enumerate(args.configs)}
        tasks.sort(key=lambda task: (seed_order[task['seed']], config_order[task['name']]))
    if args.dry_run:
        print(json.dumps([{"config": str(task["config"]), "seed": task["seed"],
                           "status": task["status"], "directory": str(task["directory"])}
                          for task in tasks], indent=2))
        return
    aggregate(tasks, output)
    if args.aggregate_only:
        return
    pending = [task for task in tasks if task["status"] != "complete"]
    slots = [(str(gpu), slot) for gpu in args.gpus for slot in range(args.jobs_per_gpu)]
    running = []
    try:
        while pending or running:
            for entry in list(running):
                returncode = entry["process"].poll()
                if returncode is None:
                    continue
                entry["log"].close()
                task = entry["task"]
                result = read_json(task["directory"] / "result.json")
                task["status"] = "complete" if returncode == 0 and result and result.get("status") == "complete" else "failed"
                if task["status"] == "complete":
                    print(f"DONE {task['name']} seed={task['seed']} test={result['test_accuracy']:.6f}", flush=True)
                else:
                    print(f"FAILED {task['name']} seed={task['seed']} exit={returncode}; see {task['directory'] / 'run.log'}", flush=True)
                slots.append(entry["slot"])
                running.remove(entry)
                aggregate(tasks, output)
            while pending and slots:
                task = pending.pop(0)
                gpu_slot = slots.pop(0)
                directory = task["directory"]
                if (directory / "config.json").exists():
                    # Preserve prior incomplete/failed artifacts instead of overwriting.
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    directory.rename(directory.with_name(directory.name + ".previous_" + stamp))
                directory.mkdir(parents=True, exist_ok=True)
                command = [args.python, "-u", str(ROOT / "experiments/run_finger.py"),
                           "--config", str(task["config"]), "--seed", str(task["seed"]),
                           "--output", str(directory), "--threads", str(args.threads)]
                if args.paper_protocol:
                    command.append("--paper-protocol")
                if not args.save_checkpoints:
                    command.append("--no-checkpoints")
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=gpu_slot[0], PYTHONHASHSEED=str(task["seed"]),
                                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                                   OPENBLAS_NUM_THREADS=str(args.threads))
                log = (directory / "run.log").open("w", buffering=1)
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                write_json(directory / "launcher_task.json", {
                    "config": str(task["config"]), "original_config": str(task["original_config"]),
                    "input_config_sha256": task["fingerprint"],
                    "seed": task["seed"], "gpu": gpu_slot[0], "gpu_slot": gpu_slot[1],
                    "pid": process.pid, "command": command,
                    "started_utc": datetime.now(timezone.utc).isoformat()})
                task["status"] = "running"
                running.append({"task": task, "process": process, "log": log, "slot": gpu_slot})
                print(f"START {task['name']} seed={task['seed']} gpu={gpu_slot[0]} slot={gpu_slot[1]} pid={process.pid}", flush=True)
                aggregate(tasks, output)
            if pending or running:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopping active experiment processes; completed runs will be retained.", flush=True)
        for entry in running:
            entry["process"].terminate()
        for entry in running:
            try:
                entry["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                entry["process"].kill()
                entry["process"].wait()
            entry["log"].close()
            entry["task"]["status"] = "interrupted"
        aggregate(tasks, output)
        raise SystemExit(130)
    summary = aggregate(tasks, output)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
