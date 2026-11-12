#!/usr/bin/env python3
"""Train Hi-Time on FingerMovements."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

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

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare_data(config, output):
    if not config.training.merge_resplit:
        raise ValueError('FingerMovements uses the merged, stratified 60/20/20 split')
    train_x, train_y, test_x, test_y = load_uea_dataset('FingerMovements', config.data.data_dir)
    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    if x.ndim != 3 or x.shape[1:] != (28, 50):
        raise ValueError(f'Expected FingerMovements in (N,28,50) format, received {x.shape}')
    indices = np.arange(len(x))
    train_indices, temporary = train_test_split(
        indices, test_size=0.4, random_state=config.training.seed, stratify=y)
    val_indices, test_indices = train_test_split(
        temporary, test_size=0.5, random_state=config.training.seed, stratify=y[temporary])
    mean = x[train_indices].mean(axis=(0, 2), keepdims=True)
    std = x[train_indices].std(axis=(0, 2), keepdims=True) + 1e-8
    np.savez_compressed(output / 'split_and_normalization.npz',
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


@torch.no_grad()
def predict(model, loader, device, stage1=False):
    model.eval()
    all_labels, all_probabilities = [], []
    for x, y in loader:
        outputs = model(x.to(device))
        logits = outputs['logits' if stage1 else 'cls_logits']
        all_labels.append(y.numpy())
        all_probabilities.append(logits.softmax(dim=-1).float().cpu().numpy())
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=-1)
    return labels, probabilities, predictions, float(np.mean(predictions == labels))


def train_one(args):
    output = args.output / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() or (output / 'results.json').exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    config = Config.load(args.config)
    if config.data.dataset_name != 'FingerMovements':
        raise ValueError('Only FingerMovements is supported')
    config.training.seed = args.seed
    config.training.num_runs = 1
    config.checkpoint_dir = str(output / 'checkpoints')
    config.data.data_dir = str((ROOT / config.data.data_dir).resolve())
    if (ROOT / config.model.gpt2_model).is_dir():
        config.model.gpt2_model = str((ROOT / config.model.gpt2_model).resolve())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    train_loader, val_loader, test_loader, test_indices = prepare_data(config, output)
    write_json(output / 'config.json', asdict(config))
    name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'Seed {args.seed}, device: {name}', flush=True)
    start = time.monotonic()
    result = {'seed': args.seed, 'status': 'running', 'stages': {},
              'split_sizes': {'train': len(train_loader.dataset), 'val': len(val_loader.dataset),
                              'test': len(test_loader.dataset)}}
    write_json(output / 'results.json', result)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)

    def save_model(name, model):
        torch.save({'model_state_dict': model.state_dict(), 'config': asdict(config),
                    'seed': args.seed}, checkpoints / f'{name}.pt')

    try:
        stage_start = time.monotonic()
        print('Stage 1: encoder', flush=True)
        stage1 = train_stage1(config, train_loader, val_loader, device)
        history = {'stage1': dict(stage1.metric_tracker.history)}
        write_json(output / 'history.json', history)
        save_model('encoder', stage1.model)
        result['stages']['stage1'] = {'seconds': time.monotonic() - stage_start,
                                      'best_val_accuracy': stage1.early_stopping.best_score}
        write_json(output / 'results.json', result)

        # Reset each stage independently, matching the original training sequence.
        set_seed(args.seed)
        stage_start = time.monotonic()
        print('Stage 2: residual K-Means', flush=True)
        discretizer = train_stage2(config, stage1.model, train_loader, val_loader, device)
        save_model('discretizer', discretizer)
        result['stages']['stage2'] = {'seconds': time.monotonic() - stage_start}
        write_json(output / 'results.json', result)

        set_seed(args.seed)
        stage_start = time.monotonic()
        print('Stage 3: GPT-2 predictor', flush=True)
        stage3 = train_stage3(config, stage1.model, discretizer, train_loader, val_loader, device)
        history['stage3'] = dict(stage3.metric_tracker.history)
        write_json(output / 'history.json', history)
        save_model('predictor', stage3.model)
        result['stages']['stage3'] = {'seconds': time.monotonic() - stage_start,
                                      'best_val_accuracy': stage3.early_stopping.best_score}
        labels, probabilities, predictions, test_accuracy = predict(stage3.model, test_loader, device)
        _, _, _, val_accuracy = predict(stage3.model, val_loader, device)
        _, stage1_probabilities, stage1_predictions, stage1_accuracy = predict(
            stage1.model, test_loader, device, stage1=True)
        np.savez_compressed(output / 'test_predictions.npz', labels=labels, probabilities=probabilities,
                            predictions=predictions, test_indices=test_indices,
                            stage1_probabilities=stage1_probabilities, stage1_predictions=stage1_predictions)
        result.update(status='complete', test_accuracy=test_accuracy,
                      test_correct=int((predictions == labels).sum()), test_total=len(labels),
                      validation_accuracy=val_accuracy, stage1_test_accuracy=stage1_accuracy,
                      elapsed_seconds=time.monotonic() - start)
    except BaseException as error:
        result.update(status='failed', error=str(error), elapsed_seconds=time.monotonic() - start)
        traceback.print_exc()
        raise
    finally:
        write_json(output / 'results.json', result)


def launch(args, seeds):
    if Config.load(args.config).data.dataset_name != 'FingerMovements':
        raise ValueError('Only FingerMovements is supported')
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        output = args.output / f'seed_{seed}'
        if (output / 'config.json').exists() or (output / 'results.json').exists():
            raise FileExistsError(f'Choose a new output directory: {output}')
    pending, free_gpus, running = list(seeds), list(args.gpus), []
    states = {seed: {'seed': seed, 'status': 'pending'} for seed in seeds}

    def summarize():
        runs = [states[seed] for seed in seeds]
        accuracies = [run['test_accuracy'] for run in runs if run['status'] == 'complete']
        summary = {'runs': runs, 'completed': len(accuracies), 'requested': len(seeds),
                   'mean_accuracy': float(np.mean(accuracies)) if len(accuracies) == len(seeds) else None,
                   'std_accuracy': float(np.std(accuracies)) if len(accuracies) == len(seeds) else None}
        write_json(args.output / 'results.json', summary)
        return summary

    summarize()
    try:
        while pending or running:
            for entry in list(running):
                process, seed, gpu, log = entry
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                path = args.output / f'seed_{seed}' / 'results.json'
                result = json.loads(path.read_text()) if path.exists() else {'seed': seed}
                if returncode == 0 and result.get('status') == 'complete':
                    print(f"Seed {seed}: {result['test_correct']}/{result['test_total']} "
                          f"({result['test_accuracy']:.4%})", flush=True)
                else:
                    result.update(status='failed', returncode=returncode)
                    print(f'Seed {seed} failed; see {path.parent / "train.log"}', flush=True)
                states[seed] = result
                free_gpus.append(gpu)
                running.remove(entry)
                summarize()
            while pending and free_gpus:
                seed, gpu = pending.pop(0), free_gpus.pop(0)
                output = args.output / f'seed_{seed}'
                output.mkdir(exist_ok=True)
                log = (output / 'train.log').open('w', buffering=1)
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed),
                                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                                   OPENBLAS_NUM_THREADS=str(args.threads))
                command = [sys.executable, '-u', str(ROOT / 'train.py'), '--worker',
                           '--seed', str(seed), '--config', str(args.config),
                           '--output', str(args.output), '--threads', str(args.threads)]
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                running.append((process, seed, gpu, log))
                states[seed] = {'seed': seed, 'status': 'running'}
                print(f'Starting seed {seed} on GPU {gpu}; log: {output / "train.log"}', flush=True)
            if running:
                time.sleep(1)
    except BaseException:
        for process, seed, _, log in running:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            states[seed] = {'seed': seed, 'status': 'interrupted'}
        summarize()
        raise
    summary = summarize()
    if summary['completed'] != len(seeds):
        raise SystemExit(1)
    print(f"Mean accuracy: {summary['mean_accuracy']:.4%} "
          f"(std: {summary['std_accuracy']:.4%})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/fingermovements.json'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed', type=int, help='Run one seed')
    group.add_argument('--seeds', type=int, nargs='+', help='Run the supplied seeds')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='One worker per GPU; default: 0')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error('Seeds must be distinct integers in [0, 2**32)')
    if args.threads < 1 or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error('Threads must be positive and GPU IDs must be distinct nonnegative integers')
    args.config = (ROOT / args.config).resolve()
    args.output = (ROOT / args.output).resolve()
    if args.worker:
        if args.seed is None:
            parser.error('--worker requires --seed')
        train_one(args)
    else:
        launch(args, seeds)


if __name__ == '__main__':
    main()
