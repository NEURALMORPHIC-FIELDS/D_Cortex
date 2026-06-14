# Local single-GPU training (Windows, NVIDIA RTX 5080 / Blackwell sm_120)

This documents the local adaptation of `colab/step2_training_v11.py` for a single
NVIDIA GPU on Windows. The model architecture under `dcortex/` is imported as-is
and is not modified. The adaptation lives in three files:

| File | Role |
|------|------|
| `scripts/verify_env_local.py` | Gate 1 (ENV): CUDA available, compute capability (12,0), a real bf16 cuda matmul. |
| `colab/train_local.py` | The training run: dual-agent structural episodes + a local-corpus LM slice. |
| `scripts/emit_verdict.py` | Reads the run artifacts and evaluates four gates (ENV/FIT/LEARNS/RESUME), writes `verdict.json` and `loss_curve.png`. |

## Hardware target

- Blackwell sm_120 (RTX 5080, 16 GB) requires **PyTorch >= 2.7 built for CUDA 12.8
  (cu128)** plus a recent NVIDIA driver. Older cu121/cu124 wheels fail with
  `no kernel image available for execution on the device`.
- Precision: **bf16 autocast, NO GradScaler** (Blackwell takes the A100 code path,
  selected automatically when compute capability major >= 8).
- 16 GB VRAM: gradient checkpointing is enabled on the decoder standard blocks and
  the batch is small; the measured peak is well under the budget.

## Setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install tiktoken "datasets>=2.14" numpy matplotlib
```

## Run

```powershell
# Gate 1: environment
.\.venv\Scripts\python.exe scripts\verify_env_local.py

# Training (300 steps, fresh). Override the LM corpus with --lm-source.
.\.venv\Scripts\python.exe colab\train_local.py --fresh

# Evaluate the four gates, write verdict.json and loss_curve.png
.\.venv\Scripts\python.exe scripts\emit_verdict.py
```

Artifacts are written under `runs/dcortex/` (git-ignored): `checkpoints/` (atomic,
numeric-sort resumable), `results/loss_history.json`, `results/loss_curve.png`,
`results/run_meta_main.json`, `results/verdict.json`.

## Notes

- The LM slice (25 percent of micro-steps) is fed from a local JSONL corpus with a
  `"text"` field; set the path with `--lm-source`. The structural episodes (the
  Objective B1 task) are synthetic and need no external data.
- Checkpoints persist model, optimizer, step, epoch, loss history, config and RNG
  state. Resume is automatic from the highest-numbered checkpoint
  (`int(name.split("_step_")[1].split(".")[0])`); restoring RNG makes the resumed
  trajectory bit-identical to an uninterrupted run.
- Reference run on an RTX 5080 (300 steps): aggregate train loss fell from about
  11.79 to 3.08, peak allocated VRAM about 3.23 GB, no OOM, in roughly 7.8 minutes.
