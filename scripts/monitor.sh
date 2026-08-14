#!/usr/bin/env bash
set -u
ROOT=${REPRO_ROOT:-/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5}
TMUX=${TMUX_BIN:-/home/ai/.local/opt/tmux-3.0a/usr/bin/tmux}
echo "time=$(date --iso-8601=seconds) host=$(hostname)"
if "$TMUX" has-session -t codex-limit-rlvr-qwen7b-math500-v5 2>/dev/null; then
    echo 'tmux=running'
else
    echo 'tmux=absent'
fi
echo 'status_markers:'
find "$ROOT/status" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort
echo 'pipeline_tail:'
tail -30 "$ROOT/status/pipeline.log" 2>/dev/null || true
echo 'result_files:'
find "$ROOT/results" -type f \( -name '*.jsonl' -o -name '*_metrics.json' -o -name 'VALIDATION.json' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
echo 'gpu:'
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
echo 'disk:'
df -h /home | tail -1
du -sh "$ROOT" 2>/dev/null || true
