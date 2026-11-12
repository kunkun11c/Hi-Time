#!/usr/bin/env bash
# Isolate torch's shared libraries from host images that prepend another torch.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
environment_path="${HITIME_PAPER_ENV:-$project_root/.venv-torch20}"
if [[ ! -x "$environment_path/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "$environment_path" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$environment_path/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$environment_path/bin/python" "$@"
