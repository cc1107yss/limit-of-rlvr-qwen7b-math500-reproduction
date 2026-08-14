#!/usr/bin/env python3
"""Independent pooled pass@k and paired audit for the completed 8-run study."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


ROOT = Path(
    os.environ.get(
        "REPRO_ROOT",
        "/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5",
    )
)
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
LABELS = ("base", "simplerl")
SEEDS = (1, 2, 3, 4)
KS = (1, 2, 4, 8, 16, 32, 64, 128)
REQUIRED_NON_NULL_KEYS = ("code", "score", "finish_reason")


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def load_run(label: str, seed: int) -> dict[int, dict]:
    files = sorted((RESULTS / label / f"seed-{seed}" / "math500").glob("*.jsonl"))
    if len(files) != 1:
        raise AssertionError(f"{label} seed {seed}: expected one JSONL, found {files}")
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    if len(rows) != 500 or sorted(row["idx"] for row in rows) != list(range(500)):
        raise AssertionError(f"{label} seed {seed}: invalid row/index structure")
    for row in rows:
        for key in ("code", "pred", "score", "finish_reason"):
            if not isinstance(row.get(key), list) or len(row[key]) != 32:
                raise AssertionError(f"{label} seed {seed} idx {row['idx']}: invalid {key}")
        for key in REQUIRED_NON_NULL_KEYS:
            if any(value is None for value in row[key]):
                raise AssertionError(f"{label} seed {seed} idx {row['idx']}: invalid {key}")
    return {row["idx"]: row for row in rows}


def bootstrap_ci(differences: list[float], repetitions: int = 20000) -> tuple[float, float]:
    rng = random.Random(250413837)
    n = len(differences)
    draws = sorted(mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(repetitions))
    return draws[int(0.025 * repetitions)], draws[int(0.975 * repetitions)]


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return float(ordered[low])
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        "mean": mean(ordered),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": max(ordered),
    }


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    runs = {(label, seed): load_run(label, seed) for label in LABELS for seed in SEEDS}
    model_manifest = json.loads((ROOT / "manifests" / "models.json").read_text())["models"]
    model_data: dict[str, dict] = {}
    curve_rows: list[dict[str, object]] = []
    for label in LABELS:
        per_problem: dict[int, dict[str, object]] = {}
        all_chars: list[int] = []
        all_codes: list[str] = []
        finish_counts: Counter[str] = Counter()
        empty_outputs = 0
        empty_predictions = 0
        prediction_parse_failures = 0
        seed_metrics: dict[str, dict[str, float]] = {}
        for seed in SEEDS:
            seed_counts = [sum(bool(x) for x in runs[label, seed][idx]["score"]) for idx in range(500)]
            seed_metrics[str(seed)] = {
                str(k): mean(pass_at_k(32, c, k) for c in seed_counts)
                for k in KS if k <= 32
            }
        for idx in range(500):
            scores: list[bool] = []
            for seed in SEEDS:
                row = runs[label, seed][idx]
                scores.extend(bool(value) for value in row["score"])
                all_chars.extend(len(value) for value in row["code"])
                all_codes.extend(row["code"])
                empty_outputs += sum(not isinstance(value, str) or not value.strip() for value in row["code"])
                empty_predictions += sum(not isinstance(value, str) or not value.strip() for value in row["pred"])
                prediction_parse_failures += sum(value is None for value in row["pred"])
                finish_counts.update(str(value) for value in row["finish_reason"])
            c = sum(scores)
            per_problem[idx] = {
                "correct": c,
                "pass": {str(k): pass_at_k(128, c, k) for k in KS},
                "solved_at_128": c > 0,
            }
        pooled = {str(k): mean(per_problem[idx]["pass"][str(k)] for idx in range(500)) for k in KS}
        tokenizer = AutoTokenizer.from_pretrained(
            model_manifest[label]["snapshot_path"], trust_remote_code=True, local_files_only=True
        )
        token_lengths: list[int] = []
        for start in range(0, len(all_codes), 128):
            encoded = tokenizer(
                all_codes[start : start + 128],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            token_lengths.extend(len(ids) for ids in encoded["input_ids"])
        for k in KS:
            curve_rows.append({"model": label, "k": k, "pass_at_k": pooled[str(k)]})
        model_data[label] = {
            "seed_pass_at_k": seed_metrics,
            "pooled_pass_at_k": pooled,
            "correct_generations": sum(int(per_problem[idx]["correct"]) for idx in range(500)),
            "solved_at_128": sum(bool(per_problem[idx]["solved_at_128"]) for idx in range(500)),
            "empty_outputs": empty_outputs,
            "empty_predictions": empty_predictions,
            "prediction_parse_failures": prediction_parse_failures,
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "length_truncated": finish_counts.get("length", 0),
            "output_chars": distribution(all_chars),
            "output_tokens": distribution(token_lengths),
            "per_problem": per_problem,
        }

    pairwise: dict[str, object] = {}
    for k in KS:
        differences = [
            model_data["simplerl"]["per_problem"][idx]["pass"][str(k)]
            - model_data["base"]["per_problem"][idx]["pass"][str(k)]
            for idx in range(500)
        ]
        ci_low, ci_high = bootstrap_ci(differences)
        pairwise[str(k)] = {
            "simplerl_minus_base": mean(differences),
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "simplerl_wins": sum(value > 1e-15 for value in differences),
            "base_wins": sum(value < -1e-15 for value in differences),
            "ties": sum(abs(value) <= 1e-15 for value in differences),
        }
    base_solved = [bool(model_data["base"]["per_problem"][idx]["solved_at_128"]) for idx in range(500)]
    rl_solved = [bool(model_data["simplerl"]["per_problem"][idx]["solved_at_128"]) for idx in range(500)]
    categories = {
        "both": sum(a and b for a, b in zip(base_solved, rl_solved)),
        "base_only": sum(a and not b for a, b in zip(base_solved, rl_solved)),
        "simplerl_only": sum(not a and b for a, b in zip(base_solved, rl_solved)),
        "neither": sum(not a and not b for a, b in zip(base_solved, rl_solved)),
    }
    pairwise["pass128_categories"] = categories

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "1 - C(n-c,k) / C(n,k)",
        "n_per_problem": 128,
        "validation_policy": "pred=None is counted as an observed parse failure; code/score/finish_reason must be non-null",
        "models": model_data,
        "paired": pairwise,
        "paper_table2_reference": {
            "base_pass128": 0.960,
            "simplerl_pass128": 0.934,
            "both": 462,
            "base_only": 18,
            "simplerl_only": 5,
            "neither": 15,
        },
    }
    (AUDIT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (AUDIT / "pass_at_k_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "k", "pass_at_k"))
        writer.writeheader()
        writer.writerows(curve_rows)

    report = [
        "# Qwen2.5-7B × MATH500 Reproduction Audit",
        "",
        f"Generated at (UTC): {result['created_utc']}",
        "",
        "## Experimental setup",
        "",
        "Official code commit `79c348f4543330bb78b01a5332df09fea2700f70`; 500 MATH500 problems; official zero-shot `qwen-boxed` prompt without an additional chat template. temperature=0.6, top-p=0.95, maximum generation length=16,384, maximum model length=32,768, tensor parallelism=1, GPU memory utilization=0.9, and the default dtype. Each model uses seeds 1–4 with 32 samples per problem per seed, for 128 observed samples per problem.",
        "",
        "## Pooled pass@k (128 observed samples per problem across four seeds)",
        "",
        "| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        values = model_data[label]["pooled_pass_at_k"]
        report.append("| " + label + " | " + " | ".join(f"{100 * values[str(k)]:.2f}%" for k in KS) + " |")
    report += [
        "",
        "## Problem-level solvability at pass@128",
        "",
        f"Solved by both: {categories['both']}/500; Base only: {categories['base_only']}/500; SimpleRL only: {categories['simplerl_only']}/500; solved by neither: {categories['neither']}/500.",
        "",
        f"The SimpleRL minus Base pass@128 difference is {100 * pairwise['128']['simplerl_minus_base']:.2f} percentage points, with a problem-paired bootstrap 95% CI of [{100 * pairwise['128']['paired_bootstrap_95_ci'][0]:.2f}, {100 * pairwise['128']['paired_bootstrap_95_ci'][1]:.2f}] percentage points.",
        "",
        "`pred=None` is retained as an observed parse failure and counted as an incorrect sample; it does not indicate a missing generation, score, or finish reason.",
        "",
        "Complete machine-readable results are in `audit/results.json`; the pass@k curve is in `audit/pass_at_k_curve.csv`.",
    ]
    (AUDIT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"models": {k: v["pooled_pass_at_k"] for k, v in model_data.items()}, "categories": categories}, indent=2))


if __name__ == "__main__":
    main()
