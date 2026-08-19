# Limit of RLVR: Qwen2.5-7B × MATH500 Reproduction

This repository contains an auditable reproduction of the Qwen2.5-7B MATH500 comparison in Figure 2 and Table 2 of [*Limit of Reinforcement Learning with Verifiable Rewards* (arXiv:2504.13837 v5)](https://arxiv.org/abs/2504.13837).

The reproduction evaluates the authors' public Base and SimpleRL checkpoints. It does **not** retrain GRPO. The official evaluator, prompt, immutable model revisions, dataset, sampling configuration, and eight-run execution order are fixed and documented below.

> **Supervisor-ready experiment report:** [English](reports/EXPERIMENT_REPORT.en.md) | [中文](reports/EXPERIMENT_REPORT.zh-CN.md)

## Main result

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B Base | 61.39% | 73.41% | 81.44% | 86.82% | 90.43% | 92.93% | 94.74% | **96.20%** |
| Qwen2.5-7B SimpleRL-Zoo | **77.93%** | **82.36%** | **85.68%** | **88.30%** | **90.22%** | **91.51%** | **92.35%** | 93.00% |

Each pass@k value is computed from the 128 observed samples for every problem using

\[
\operatorname{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\]

where \(n=128\) and \(c\) is the number of correct samples for that problem. Values are then averaged across the 500 problems.

At pass@128, SimpleRL minus Base is **−3.20 percentage points**, with a problem-paired bootstrap 95% confidence interval of **[−5.00, −1.40]** points. SimpleRL wins on 3 problems, Base wins on 19, and 478 are ties. This reproduces the paper's central high-budget finding: RLVR substantially improves pass@1 but can reduce the set of problems solved under large sampling budgets.

## Exact evaluation setup

| Component | Fixed value |
|---|---|
| Paper | arXiv:2504.13837 v5, Figure 2 / Table 2 |
| Official repository | `LeapLabTHU/limit-of-RLVR@79c348f4543330bb78b01a5332df09fea2700f70` |
| Base checkpoint | `Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796` |
| RLVR checkpoint | `hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo@d630142f26acc8adf8051298cba8023232169d56` |
| Dataset | Repository MATH500, 500 problems, SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| Prompt | Official zero-shot `qwen-boxed`; no additional chat template |
| Sampling | temperature `0.6`, top-p `0.95`, seeds `1–4`, 32 samples per problem per seed |
| Length | maximum generation `16,384`; maximum model length `32,768` |
| vLLM | tensor parallel `1`, GPU memory utilization `0.9`, default dtype |
| Total outputs | 64,000 per model; 128,000 overall |
| Run order | Base seeds 1–4, then SimpleRL seeds 1–4 |

No output was selectively rerun because of numerical disagreement with the paper. The recovery history and transport-only accommodations are documented in [PROTOCOL.md](PROTOCOL.md).

## Comparison with the paper

| Metric | Paper | This reproduction |
|---|---:|---:|
| Base pass@128 | 96.0% | 96.2% |
| SimpleRL pass@128 | 93.4% | 93.0% |
| Solved by both | 462 | 462 |
| Base only | 18 | 19 |
| SimpleRL only | 5 | 3 |
| Solved by neither | 15 | 16 |

The small differences are consistent with stochastic sampling. Configuration fidelity and numerical identity are reported separately; no result-selection criterion was used.

## Repository contents

- `scripts/`: launch, download, validation, monitoring, environment capture, and independent pooled-audit code.
- `audit/results.json`: complete machine-readable aggregate and per-problem statistics.
- `audit/pass_at_k_curve.csv`: pooled pass@k curve for both models.
- `reports/REPORT.md`: concise English audit report.
- `environment/`: captured core software versions, wheel hashes, GPU, driver, disk, and orchestration provenance records.
- `manifests/raw-results.sha256`: hashes of the eight original JSONL result files.
- `PROTOCOL.md`: execution protocol, recovery decisions, and known reproduction limitations.

Model weights and raw generations are intentionally not committed. The checkpoints are publicly available at immutable revisions, while the eight raw result files total approximately 243 MB and are identified by path, size, and SHA-256 in the manifests.

## Reproduce the evaluation

1. Clone the official code and check out the exact commit listed above.
2. Build a Python 3.10 environment following the pinned core package versions and captured environment records.
3. Download both immutable model revisions with `scripts/download_models.py`.
4. Review and run `scripts/run_pipeline.sh` from the isolated reproduction root.
5. Validate every run with `scripts/validate_run.py`.
6. Set `REPRO_ROOT` if the reproduction root differs from the original server path, then run `scripts/audit_all.py`.

The orchestration scripts intentionally encode the audited server layout. Review paths before launch; changing paths does not change the evaluation configuration.

## Validation policy

Every result file must contain exactly 500 unique indices and 32 entries each for generation, prediction, score, and finish reason. `code`, `score`, and `finish_reason` must be non-null. An observed `pred=None` is retained as a parse failure and counted as incorrect rather than treated as a missing run.

Base produced six empty generations and six corresponding parse failures across 64,000 samples. SimpleRL produced no empty generations and one length-truncated sample. All 128,000 generations retained non-null score and finish-reason data.

## Citation and attribution

If this repository is useful, cite the original paper and repository:

- Paper: [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)
- Official code: [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR)
- Base checkpoint: [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- RLVR checkpoint: [hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo)

No upstream source code or model weights are vendored here. Consult the upstream projects for their respective license terms.
