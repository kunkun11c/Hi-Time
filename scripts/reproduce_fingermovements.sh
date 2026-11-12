#!/usr/bin/env bash
# Run the recorded FingerMovements configuration and five seeds.
# Usage: HITIME_GPUS="0 1 2 3 4" scripts/reproduce_fingermovements.sh [output_directory]
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="${1:-$project_root/experiments/results/fingermovements}"
if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [output_directory]\n' "$0" >&2
  exit 2
fi
read -r -a gpu_ids <<< "${HITIME_GPUS:-0 1 2 3 4}"
seeds=(957 1291 1491 1903 1498)
cd "$project_root"
exec "$project_root/scripts/python_paper.sh" -u "$project_root/experiments/launch_search.py" \
  --python "$project_root/scripts/python_paper.sh" \
  --configs "$project_root/configs/fingermovements.json" \
  --seeds "${seeds[@]}" \
  --gpus "${gpu_ids[@]}" \
  --threads 4 --jobs-per-gpu 1 \
  --paper-protocol --save-checkpoints \
  --output-root "$output_directory"
