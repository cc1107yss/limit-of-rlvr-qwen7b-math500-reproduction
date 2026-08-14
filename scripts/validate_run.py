#!/usr/bin/env python3
"""Strict structural validation for one official math_eval JSONL result."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ARRAY_KEYS = ("code", "pred", "score", "finish_reason")
# The evaluator deliberately uses ``None`` when an otherwise valid generation
# has no parseable final answer.  That is a scored failure, not malformed
# output.  All other arrays must still be fully populated.
REQUIRED_NON_NULL_KEYS = ("code", "score", "finish_reason")


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--n-sampling", default=32, type=int)
    args = ap.parse_args()

    files = sorted(args.output_dir.glob("math500/*.jsonl"))
    if len(files) != 1:
        raise AssertionError(f"Expected exactly one JSONL under {args.output_dir}, found {files}")
    path = files[0]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise AssertionError(f"Expected 500 rows, found {len(rows)}")
    indices = [row.get("idx") for row in rows]
    if len(set(indices)) != 500:
        raise AssertionError("Indices are not unique")
    expected = list(range(500))
    if sorted(indices) != expected:
        raise AssertionError(f"Indices are incomplete: got min/max {min(indices)}/{max(indices)}")

    none_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    char_lengths: list[int] = []
    correct_per_problem: list[int] = []
    empty_outputs = 0
    empty_predictions = 0
    for row in rows:
        for key in ARRAY_KEYS:
            if key not in row:
                raise AssertionError(f"idx={row['idx']} missing {key}")
            values = row[key]
            if not isinstance(values, list) or len(values) != args.n_sampling:
                raise AssertionError(f"idx={row['idx']} key={key} has invalid length/type")
            none_counts[key] += sum(value is None for value in values)
        codes = row["code"]
        preds = row["pred"]
        scores = row["score"]
        finishes = row["finish_reason"]
        empty_outputs += sum(not isinstance(value, str) or not value.strip() for value in codes)
        empty_predictions += sum(not isinstance(value, str) or not value.strip() for value in preds)
        char_lengths.extend(len(value) if isinstance(value, str) else 0 for value in codes)
        correct_per_problem.append(sum(bool(value) for value in scores))
        finish_counts.update(str(value) for value in finishes)

    critical_none = sum(none_counts[key] for key in REQUIRED_NON_NULL_KEYS)
    if critical_none:
        raise AssertionError(f"Required arrays contain None: {dict(none_counts)}")
    ks = (1, 2, 4, 8, 16, 32)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "seed": args.seed,
        "jsonl": str(path),
        "rows": len(rows),
        "indices_complete_0_499": True,
        "samples_per_problem": args.n_sampling,
        "total_generations": len(rows) * args.n_sampling,
        "critical_none_counts": {key: none_counts[key] for key in ARRAY_KEYS},
        "prediction_parse_failures": none_counts["pred"],
        "validation_policy": "pred=None is an observed parse failure; code/score/finish_reason must be non-null",
        "correct_generations": sum(correct_per_problem),
        "pass_at_k": {
            str(k): mean(pass_at_k(args.n_sampling, c, k) for c in correct_per_problem)
            for k in ks
        },
        "empty_outputs": empty_outputs,
        "empty_predictions": empty_predictions,
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "length_truncated": finish_counts.get("length", 0),
        "output_chars": {
            "mean": mean(char_lengths),
            "p50": percentile(char_lengths, 0.50),
            "p90": percentile(char_lengths, 0.90),
            "p95": percentile(char_lengths, 0.95),
            "p99": percentile(char_lengths, 0.99),
            "max": max(char_lengths),
        },
    }
    out = args.output_dir / "VALIDATION.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
