#!/usr/bin/env bash
# Build an isolated reproduction environment without changing the host Python.
# torch version/CUDA source: https://pytorch.org/get-started/previous-versions/#v200
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
environment_path="${HITIME_PAPER_ENV:-$project_root/.venv-torch20}"
package_index="${HITIME_PACKAGE_INDEX:-https://pypi.org/simple}"
if [[ ! -x "$environment_path/bin/python" ]]; then
  if [[ -x "$project_root/.python39/bin/python3.9" ]]; then
    "$project_root/.python39/bin/python3.9" -m venv "$environment_path"
  else
    uv --native-tls venv --python 3.9 --managed-python "$environment_path"
  fi
fi
uv --native-tls pip install --python "$environment_path/bin/python" \
  'torch==2.0.0+cu118' --index-url https://download.pytorch.org/whl/cu118
uv --native-tls pip install --python "$environment_path/bin/python" \
  -r "$project_root/requirements-paper-lock.txt" --index-url "$package_index"
HITIME_PAPER_ENV="$environment_path" "$project_root/scripts/python_paper.sh" -c 'import sys, torch, transformers; print(sys.version); print("torch", torch.__version__, "CUDA", torch.version.cuda); print("transformers", transformers.__version__); print("CUDA available", torch.cuda.is_available())'
printf 'Environment ready: %s\n' "$environment_path"
