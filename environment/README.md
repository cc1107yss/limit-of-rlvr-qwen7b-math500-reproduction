# Captured environment

These files record the core runtime used for the completed reproduction:

- Python 3.10.20
- PyTorch 2.4.0+cu124
- vLLM 0.6.3
- Transformers 4.47.1
- Ray 2.10.0
- SymPy 1.12
- ANTLR runtime 4.11.1
- FlashAttention 2.6.3
- NVIDIA GeForce RTX 3090, 24 GiB, driver 550.120

`core-wheel-sha256.txt` identifies the retained core installation artifacts. The wheel binaries are not committed because of their size. `orchestration-sha256.txt` records the exact scripts executed on the server. The public copies under `scripts/` differ only in documentation-facing report language and the addition of optional `REPRO_ROOT`/`TMUX_BIN` path overrides; the evaluation parameters are unchanged.

Machine-specific GPU UUID and detailed host identifiers are intentionally omitted from this public repository.
