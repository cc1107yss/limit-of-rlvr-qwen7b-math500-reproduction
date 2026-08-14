#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${REPRO_ROOT:-/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5}
REPO="$ROOT/repo"
PY="$ROOT/.venv/bin/python"
ENV_DIR="$ROOT/environment"
mkdir -p "$ENV_DIR"

"$PY" -m pip freeze --all > "$ENV_DIR/pip-freeze.txt"
"$PY" -m pip inspect --local > "$ENV_DIR/pip-inspect.json"
"$PY" -m pip check > "$ENV_DIR/pip-check.txt"
"$PY" -m pip show torch vllm transformers ray sympy antlr4-python3-runtime flash-attn > "$ENV_DIR/core-packages.txt"
"$PY" - <<'PY' > "$ENV_DIR/core-versions.json"
import importlib.metadata as m
import json
import platform
import torch

names = ["torch", "vllm", "transformers", "ray", "sympy", "antlr4-python3-runtime", "flash-attn"]
print(json.dumps({
    "python": platform.python_version(),
    "packages": {name: m.version(name) for name in names},
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY
nvidia-smi -q > "$ENV_DIR/nvidia-smi-q.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$ENV_DIR/gpu.csv"
uname -a > "$ENV_DIR/uname.txt"
df -h /home > "$ENV_DIR/disk-at-start.txt"

cd "$REPO"
git show --no-patch --format=fuller HEAD > "$ENV_DIR/repo-commit.txt"
git status --porcelain=v1 --branch > "$ENV_DIR/repo-status.txt"
sha256sum \
  math/examples/math_eval/data/math500/test.jsonl \
  math/examples/math_eval/math_eval.py \
  math/examples/math_eval/utils.py \
  math/pass@k.py \
  math/requirements.txt > "$ENV_DIR/source-sha256.txt"

sha256sum "$ROOT"/scripts/* > "$ENV_DIR/orchestration-sha256.txt"
sha256sum "$ROOT"/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl > "$ENV_DIR/flash-attn-wheel-sha256.txt"
