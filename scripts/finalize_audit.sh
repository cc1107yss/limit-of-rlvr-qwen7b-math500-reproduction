#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

ROOT=${REPRO_ROOT:-/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5}
PY="$ROOT/.venv/bin/python"
STATUS_DIR="$ROOT/status"
LOG_DIR="$ROOT/logs"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$ROOT/audit"
exec > >(tee -a "$STATUS_DIR/pipeline.log") 2>&1

if [[ ! -f "$STATUS_DIR/PIPELINE_FAILED" ]] \
    || [[ "$(cat "$STATUS_DIR/FAILED_STAGE" 2>/dev/null || true)" != final_audit ]]; then
    echo "Refusing final-audit recovery without the expected final_audit marker"
    exit 2
fi

stamp=$(date +%Y%m%dT%H%M%S)
recovery="$STATUS_DIR/recovered-final-audit-$stamp"
mkdir -p "$recovery"
mv "$STATUS_DIR/PIPELINE_FAILED" "$STATUS_DIR/FAILED_STAGE" "$STATUS_DIR/FAILED_RC" "$recovery/"
printf '%s FINAL_AUDIT_RECOVERY_AUTHORIZED preserved_failure=%s\n' \
    "$(date --iso-8601=seconds)" "$recovery"

for label in base simplerl; do
    for seed in 1 2 3 4; do
        printf '%s FINAL_VALIDATE label=%s seed=%s\n' \
            "$(date --iso-8601=seconds)" "$label" "$seed"
        "$PY" "$ROOT/scripts/validate_run.py" \
            --label "$label" --seed "$seed" \
            --output-dir "$ROOT/results/$label/seed-$seed" --n-sampling 32 \
            | tee "$LOG_DIR/${label}-seed${seed}-final-validation.log"
    done
done

printf '%s FINAL_AUDIT_START\n' "$(date --iso-8601=seconds)"
"$PY" "$ROOT/scripts/audit_all.py" | tee "$LOG_DIR/final-audit.log"
touch "$STATUS_DIR/PIPELINE_COMPLETE"
printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)"
