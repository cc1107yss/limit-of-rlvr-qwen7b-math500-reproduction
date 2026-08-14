#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

ROOT=${REPRO_ROOT:-/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5}
REPO="$ROOT/repo"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
STATUS_DIR="$ROOT/status"
RESULTS="$ROOT/results"
LINK_DIR="$ROOT/runtime-model-links"
PIPELINE_LOG="$STATUS_DIR/pipeline.log"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULTS" "$LINK_DIR" "$ROOT/manifests"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

CURRENT_STAGE=bootstrap
on_error() {
    rc=$?
    printf '%s PIPELINE_FAILED stage=%s rc=%s\n' "$(date --iso-8601=seconds)" "$CURRENT_STAGE" "$rc"
    printf '%s\n' "$CURRENT_STAGE" > "$STATUS_DIR/FAILED_STAGE"
    printf '%s\n' "$rc" > "$STATUS_DIR/FAILED_RC"
    touch "$STATUS_DIR/PIPELINE_FAILED"
    exit "$rc"
}
trap on_error ERR

if [[ -e "$STATUS_DIR/PIPELINE_COMPLETE" || -e "$STATUS_DIR/PIPELINE_FAILED" ]]; then
    failed_stage="$(cat "$STATUS_DIR/FAILED_STAGE" 2>/dev/null || true)"
    recovery_kind=""
    if [[ -f "$STATUS_DIR/PIPELINE_FAILED" ]] \
        && [[ "${ALLOW_RESUME_AFTER_MODEL_DOWNLOAD_FAILURE:-0}" == 1 ]] \
        && [[ "$failed_stage" == model_download ]]; then
        recovery_kind="model-download"
    elif [[ -f "$STATUS_DIR/PIPELINE_FAILED" ]] \
        && [[ "${ALLOW_RESUME_AFTER_VALIDATION_EMPTY_PRED:-0}" == 1 ]] \
        && [[ "$failed_stage" == base_seed1 ]]; then
        recovery_kind="validation-empty-pred"
    else
        echo "Refusing to start over an existing terminal pipeline marker"
        exit 2
    fi
    RECOVERY_DIR="$STATUS_DIR/recovered-${recovery_kind}-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$RECOVERY_DIR"
    mv "$STATUS_DIR/PIPELINE_FAILED" "$STATUS_DIR/FAILED_STAGE" "$STATUS_DIR/FAILED_RC" "$RECOVERY_DIR/"
    printf '%s RECOVERY_AUTHORIZED kind=%s preserved_failure=%s\n' \
        "$(date --iso-8601=seconds)" "$recovery_kind" "$RECOVERY_DIR"
fi

export HF_HOME="$ROOT/hf-home"
export HF_HUB_CACHE="$ROOT/hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export ALL_PROXY=socks5h://127.0.0.1:11080
export HTTPS_PROXY="$ALL_PROXY"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

printf '%s PIPELINE_START host=%s\n' "$(date --iso-8601=seconds)" "$(hostname)"

cd "$REPO"
test "$(git rev-parse HEAD)" = 79c348f4543330bb78b01a5332df09fea2700f70
test -z "$(git status --porcelain)"
echo '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132  math/examples/math_eval/data/math500/test.jsonl' | sha256sum -c -
echo '2236b8688b01cb407574d84d19ee7c0c7d0704aedc237e3e4bf8a24333887560  math/examples/math_eval/math_eval.py' | sha256sum -c -
echo 'ff216eea0f23c675297898414148294ebef0314f95f4f3d91d3434311c3981cf  math/examples/math_eval/utils.py' | sha256sum -c -
echo '701bc1ddf58153c835a89b97df12fc1941fb2c1dbf3a04f9370137ab72b2ea04  math/pass@k.py' | sha256sum -c -

CURRENT_STAGE=model_download
"$PY" "$ROOT/scripts/download_models.py"

snapshot_path() {
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["snapshot_path"])' "$ROOT/manifests/models.json" "$1"
}

run_one() {
    local label=$1
    local seed=$2
    local snapshot=$3
    local link_name=$4
    local output_dir="$RESULTS/$label/seed-$seed"
    local run_log="$LOG_DIR/${label}-seed${seed}.log"
    local model_link="$LINK_DIR/$link_name"
    CURRENT_STAGE="${label}_seed${seed}"
    if find "$output_dir" -type f -print -quit 2>/dev/null | grep -q .; then
        printf '%s RUN_EXISTING_OUTPUT_VALIDATE label=%s seed=%s\n' \
            "$(date --iso-8601=seconds)" "$label" "$seed"
        "$PY" "$ROOT/scripts/validate_run.py" \
            --label "$label" --seed "$seed" --output-dir "$output_dir" --n-sampling 32
        printf '%s RUN_COMPLETE label=%s seed=%s reused_validated_output=1\n' \
            "$(date --iso-8601=seconds)" "$label" "$seed"
        return 0
    fi
    mkdir -p "$output_dir"
    rm -f "$model_link"
    ln -s "$snapshot" "$model_link"
    printf '%s RUN_START label=%s seed=%s snapshot=%s\n' "$(date --iso-8601=seconds)" "$label" "$seed" "$snapshot"
    cd "$REPO/math/examples/math_eval"
    "$PY" -u math_eval.py \
        --model_name_or_path "$model_link" \
        --data_dir "$REPO/math/examples/math_eval/data" \
        --data_names math500 \
        --output_dir "$output_dir" \
        --split test \
        --prompt_type qwen-boxed \
        --num_test_sample -1 \
        --max_tokens_per_call 16384 \
        --seed "$seed" \
        --temperature 0.6 \
        --n_sampling 32 \
        --top_p 0.95 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs 2>&1 | tee "$run_log"
    "$PY" "$ROOT/scripts/validate_run.py" \
        --label "$label" --seed "$seed" --output-dir "$output_dir" --n-sampling 32
    printf '%s RUN_COMPLETE label=%s seed=%s\n' "$(date --iso-8601=seconds)" "$label" "$seed"
}

BASE_SNAPSHOT=$(snapshot_path base)
RL_SNAPSHOT=$(snapshot_path simplerl)

for seed in 1 2 3 4; do
    run_one base "$seed" "$BASE_SNAPSHOT" Qwen2.5-7B
done
for seed in 1 2 3 4; do
    run_one simplerl "$seed" "$RL_SNAPSHOT" Qwen-2.5-7B-SimpleRL-Zoo
done

CURRENT_STAGE=final_audit
"$PY" "$ROOT/scripts/audit_all.py"
touch "$STATUS_DIR/PIPELINE_COMPLETE"
printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)"
