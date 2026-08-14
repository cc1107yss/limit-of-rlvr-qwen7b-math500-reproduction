# Qwen2.5-7B × MATH500 v5 reproduction protocol

This directory is isolated from `/home/ai/limit-of-RLVR` and evaluates public checkpoints only.

- Paper: arXiv:2504.13837 v5, Figure 2 / Table 2.
- Repository: `LeapLabTHU/limit-of-RLVR@79c348f4543330bb78b01a5332df09fea2700f70` (detached, clean).
- Data: `math500/test.jsonl`, 500 rows, SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
- Base: `Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796`.
- RLVR: `hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo@d630142f26acc8adf8051298cba8023232169d56`.
- Prompt: official `qwen-boxed`, zero shot, without `--apply_chat_template`.
- Sampling: temperature 0.6, top-p 0.95, max generation 16,384, seeds 1–4, 32 samples per seed.
- Engine: official `math_eval.py`, vLLM TP=1, `gpu_memory_utilization=0.9`, `max_model_len=32768`, default dtype.
- Order: Base seeds 1–4, then SimpleRL seeds 1–4.

The official source tree is not patched. A disposable symlink is passed as the model path because the official evaluator removes its model path after loading; deleting that symlink leaves the immutable Hugging Face cache snapshot intact.

Environment notes:

1. The historical Torch install command's cu124 index no longer serves `nvidia-cudnn-cu12==9.1.0.70`; the exact Torch 2.4.0+cu124 wheel was retained while this declared dependency was fetched from PyPI.
2. The official unpinned `pip install flash-attn --no-build-isolation` now resolves to 2.8.3.post1 and cannot build with the host's legacy CUDA 10.1 compiler. The official FlashAttention 2.6.3 prebuilt wheel for Torch 2.4, Python 3.10, CUDA 12.x, and CXX11 ABI false is installed instead. Its SHA-256 is recorded in `environment/`.
3. All other unpinned transitive packages are captured in `pip-freeze.txt`; these are an explicit public-reproduction limitation, not a claim that the authors used these later releases.
4. The server could not establish TCP 443 to Hugging Face at launch. Model bytes are therefore fetched from the same official Hugging Face URLs through an SSH reverse SOCKS tunnel on `127.0.0.1:11080`; Xet transport is disabled so the audited HTTPS proxy path is used. This changes transport only, while immutable revisions and every downloaded file hash are verified and recorded.
5. On 2026-08-13 the first SimpleRL transfer ended with an `IncompleteRead` during model download. The failed pipeline log and terminal markers were preserved under `status/recovered-model-download-*`; with explicit authorization, the identical immutable revision is resumed serially (`max_workers=1`) with transport-only retry logic before evaluation starts.
6. On 2026-08-13 Base seed 1 completed generation and official evaluation (500 × 32), with two empty generations producing `pred=None`, one evaluation timeout, and no missing `code`, `score`, or `finish_reason` entries. With explicit authorization, these two parse failures are retained as scored failures in the original JSONL. The independent validator records `pred=None` rather than treating it as a corrupt result, and the resumed pipeline validates and reuses that immutable output without regeneration or overwrite.
7. The final independent pooled audit applies the same parse-failure policy as the per-run validator: it checks every index and all 32-element arrays, requires non-null `code`, `score`, and `finish_reason`, and records `pred=None` separately as a parse failure. No original JSONL is modified or regenerated when the final audit is recovered.
8. The one-time final-audit recovery preserves the failed final-audit marker, reruns only independent per-run structural validation and `audit_all.py`, and then creates the completion marker. It never calls `math_eval.py`, loads a model, or changes any JSONL.

No stochastic deviation from the paper is a rerun criterion. If an initial run fails specifically from GPU OOM, the only allowed runtime adjustment is a documented reduction of vLLM `max_num_seqs`; no other parameter may change.
