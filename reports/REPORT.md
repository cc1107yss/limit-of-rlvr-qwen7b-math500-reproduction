# Qwen2.5-7B × MATH500 Reproduction Audit

Generated from the completed eight-run audit on 2026-08-14 UTC.

## Scope

This study reproduces the MATH500 Base-versus-SimpleRL comparison in Figure 2 and Table 2 of arXiv:2504.13837 v5. It evaluates public checkpoints only and does not retrain GRPO.

The official repository is fixed at commit `79c348f4543330bb78b01a5332df09fea2700f70`. MATH500 contains 500 problems and has SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`. The evaluation uses the official zero-shot `qwen-boxed` prompt without an additional chat template, temperature 0.6, top-p 0.95, maximum generation length 16,384, maximum model length 32,768, tensor parallelism 1, GPU memory utilization 0.9, and default dtype. Each model uses seeds 1–4 with 32 samples per problem per seed, yielding 128 samples per problem and 64,000 generations per model.

## Results

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 61.39% | 73.41% | 81.44% | 86.82% | 90.43% | 92.93% | 94.74% | **96.20%** |
| SimpleRL | **77.93%** | **82.36%** | **85.68%** | **88.30%** | **90.22%** | **91.51%** | **92.35%** | 93.00% |

At pass@128, 462 problems are solved by both models, 19 only by Base, 3 only by SimpleRL, and 16 by neither. The SimpleRL-minus-Base difference is −3.20 percentage points, with a problem-paired bootstrap 95% confidence interval of [−5.00, −1.40] points. The paired tally is 3 SimpleRL wins, 19 Base wins, and 478 ties.

The paper reports Base pass@128 of 96.0%, SimpleRL pass@128 of 93.4%, and a 462/18/5/15 split for both/Base-only/SimpleRL-only/neither. The reproduction is therefore close numerically and agrees on the main conclusion: SimpleRL improves low-budget success substantially while Base covers a broader problem set at a 128-sample budget.

## Integrity checks

All eight JSONL files passed structural validation: 500 complete indices, 32 generations per problem, and non-null generation, score, and finish-reason arrays. Base has six empty generations and six `pred=None` parse failures, retained as incorrect observations. SimpleRL has no empty generations and one output ending because of the length limit. No original JSONL was modified or selectively regenerated during final auditing.

Machine-readable statistics are in `audit/results.json`; the pass@k curve is in `audit/pass_at_k_curve.csv`; original result-file hashes are in `manifests/raw-results.sha256`.
