# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- LM_DECODER training campaign on a real local corpus.
# Built on the validated colab/train_local.py pipeline. Route: LM_DECODER
# (next-token language modeling on the decoder, Agent B). The encoder (Agent A)
# and the memory banks are NOT trained in this route: this is a deliberately
# undertrained decoder-backbone pass. The dcortex/ architecture is imported as
# is; only DCortexConfig values change (big config).
#
# Phases:
#   autotune -> probe (batch, context) under a VRAM ceiling, write autotune.json
#   train    -> tokenize corpus (hash 5% held-out, seed 42), step-0 baseline
#               perplexity, cosine-decay training with periodic held-out eval,
#               early stopping, atomic checkpoints + best_model.pt.

import argparse
import contextlib
import gc
import io
import json
import math
import os
import random
import sys
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import StandardTransformerBlock

SEP: str = "=" * 70
SEED: int = 42
SPLIT_SEED: int = 42
EVAL_SEED: int = 1234          # held-out window selection (shared with verdict)
EVAL_CONTEXT: int = 1024       # GPT-2 native context; comparable reference
DEFAULT_LM_SOURCE: str = r"E:\DATA\training_data_clean\06_FINAL_TRAINING_READY-003_BYON.jsonl"

ENC = tiktoken.get_encoding("gpt2")
EOT: int = ENC.eot_token


@contextlib.contextmanager
def silent_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# Gradient checkpointing on decoder standard blocks (training-side wrap)
# ---------------------------------------------------------------------------

_GRAD_CHECKPOINT: Dict[str, bool] = {"enabled": False}
_ORIG_STD_BLOCK_FORWARD = StandardTransformerBlock.forward


def _checkpointed_std_block_forward(self: StandardTransformerBlock, h: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if (_GRAD_CHECKPOINT["enabled"] and self.training
            and torch.is_grad_enabled() and h.requires_grad):
        return torch_checkpoint.checkpoint(_ORIG_STD_BLOCK_FORWARD, self, h,
                                           attention_mask, use_reentrant=False)
    return _ORIG_STD_BLOCK_FORWARD(self, h, attention_mask)


def enable_gradient_checkpointing() -> None:
    StandardTransformerBlock.forward = _checkpointed_std_block_forward
    _GRAD_CHECKPOINT["enabled"] = True
    print("[INFO] Gradient checkpointing ON (decoder standard blocks)", flush=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def setup_device() -> Tuple[torch.device, torch.dtype]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run scripts/verify_env_local.py.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(SEP, flush=True)
    print(f"[INFO] GPU: {name} | VRAM: {mem:.1f} GB | SM {cap[0]}.{cap[1]}", flush=True)
    if not ("A100" in name or cap[0] >= 8):
        raise RuntimeError("This campaign mandates bf16/no-GradScaler (capability >= 8).")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[INFO] bf16 autocast, TF32, NO GradScaler", flush=True)
    print(SEP, flush=True)
    return torch.device("cuda"), torch.bfloat16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def big_config() -> DCortexConfig:
    return DCortexConfig(
        hidden_dim=1024, n_enc_heads=16, n_dec_heads=16,
        enc_ff_dim=4096, dec_ff_dim=4096, n_dec_layers=16,
        n_enc_layers=4, n_fusion_layers=4, max_seq_len=2048,
    )


# ---------------------------------------------------------------------------
# Tokenization with deterministic 5% held-out split (seed 42, per-record hash)
# ---------------------------------------------------------------------------

def _is_held_out(record_index: int) -> bool:
    h = zlib.crc32(f"{SPLIT_SEED}:{record_index}".encode("utf-8"))
    return (h % 20) == 0  # ~5%


def tokenize_corpus(source_path: str, bin_dir: str, train_tokens: int,
                    val_tokens: int) -> Tuple[str, str]:
    train_path = os.path.join(bin_dir, "campaign_train.bin")
    val_path = os.path.join(bin_dir, "campaign_val.bin")
    if (os.path.exists(train_path) and os.path.getsize(train_path) >= train_tokens * 2
            and os.path.exists(val_path) and os.path.getsize(val_path) >= val_tokens * 2):
        nt = os.path.getsize(train_path) // 2
        nv = os.path.getsize(val_path) // 2
        print(f"[INFO] Corpus cached: {nt:,} train / {nv:,} val tokens", flush=True)
        return train_path, val_path
    if not os.path.exists(source_path):
        raise RuntimeError(f"Source not found: {source_path}. No synthetic fallback.")

    print(f"[INFO] Tokenizing {os.path.basename(source_path)} -> "
          f"{train_tokens:,} train / {val_tokens:,} val (gpt2, 5% held-out seed {SPLIT_SEED})",
          flush=True)
    train_buf: List[int] = []
    val_buf: List[int] = []
    record_index = 0
    lines_read = 0
    with open(source_path, "r", encoding="utf-8") as handle:
        for line in handle:
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "") or obj.get("content", "") or obj.get("story", "")
            if not text:
                continue
            held_out = _is_held_out(record_index)
            record_index += 1
            # Skip tokenizing records whose target buffer is already full.
            if held_out and len(val_buf) >= val_tokens:
                if len(train_buf) >= train_tokens:
                    break
                continue
            if (not held_out) and len(train_buf) >= train_tokens:
                if len(val_buf) >= val_tokens:
                    break
                continue
            ids = ENC.encode_ordinary(text)
            ids.append(EOT)
            (val_buf if held_out else train_buf).extend(ids)
            if len(train_buf) >= train_tokens and len(val_buf) >= val_tokens:
                break
            if lines_read % 20000 == 0:
                print(f"  {len(train_buf):,} train / {len(val_buf):,} val "
                      f"({lines_read:,} lines)", flush=True)

    if len(train_buf) < train_tokens or len(val_buf) < val_tokens:
        print(f"[WARN] Source exhausted: {len(train_buf):,} train / {len(val_buf):,} val "
              f"(requested {train_tokens:,}/{val_tokens:,}); using what is available.",
              flush=True)
    for buf, path, cap in [(train_buf, train_path, train_tokens), (val_buf, val_path, val_tokens)]:
        arr = np.array(buf[:cap], dtype=np.uint16)
        tmp = path + ".tmp"
        arr.tofile(tmp)
        os.replace(tmp, path)
    print(f"[INFO] Wrote {len(train_buf[:train_tokens]):,} train / "
          f"{len(val_buf[:val_tokens]):,} val tokens", flush=True)
    return train_path, val_path


# ---------------------------------------------------------------------------
# Batching and eval windows
# ---------------------------------------------------------------------------

def get_batch(data: np.memmap, batch: int, context: int,
              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - context - 1, size=(batch,))
    x = np.stack([data[i:i + context].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + context].astype(np.int64) for i in ix])
    xt = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
    yt = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
    return xt, yt


def eval_windows(val_data: np.memmap, n_windows: int, context: int) -> List[int]:
    rng = np.random.RandomState(EVAL_SEED)
    high = len(val_data) - context - 1
    return [int(rng.randint(0, high)) for _ in range(n_windows)]


@torch.no_grad()
def eval_perplexity(model: DCortexV2Model, val_data: np.memmap, windows: List[int],
                    context: int, device: torch.device, dtype: torch.dtype,
                    vocab: int, eval_batch: int = 8) -> Tuple[float, float]:
    model.eval()
    total_ce = 0.0
    total_tok = 0
    for b0 in range(0, len(windows), eval_batch):
        idx = windows[b0:b0 + eval_batch]
        x = np.stack([val_data[i:i + context].astype(np.int64) for i in idx])
        y = np.stack([val_data[i + 1:i + 1 + context].astype(np.int64) for i in idx])
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with silent_stdout():
            model.reset_memory()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model.decode(xt)
        ce = F.cross_entropy(logits.view(-1, vocab).float(), yt.view(-1), reduction="sum")
        total_ce += ce.item()
        total_tok += yt.numel()
    model.train()
    mean_ce = total_ce / max(1, total_tok)
    return mean_ce, math.exp(mean_ce)


# ---------------------------------------------------------------------------
# Optimizer (LM_DECODER: train decoder + shared, freeze encoder + structural heads)
# ---------------------------------------------------------------------------

def build_lm_optimizer(model: DCortexV2Model, lr: float, weight_decay: float
                       ) -> torch.optim.Optimizer:
    frozen = 0
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if (name.startswith("encoder.") or name.startswith("aux_answer_head")
                or name.startswith("value_to_key_proj")):
            p.requires_grad_(False)
            frozen += p.numel()
            continue
        (no_decay if p.dim() < 2 or "norm" in name or "bias" in name else decay).append(p)
    trained = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
    print(f"[INFO] LM_DECODER optimizer: trained={trained/1e6:.2f}M  frozen(encoder+structural)="
          f"{frozen/1e6:.2f}M", flush=True)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))


def cosine_lr(step: int, lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(path: str, model: DCortexV2Model, optimizer: torch.optim.Optimizer,
              step: int, epoch: int, loss_history: List, eval_history: List,
              best_ppl: float, model_cfg: DCortexConfig, meta: Dict) -> None:
    ckpt = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "epoch": epoch, "loss_history": loss_history,
        "eval_history": eval_history, "best_ppl": best_ppl,
        "config_model": asdict(model_cfg), "meta": meta,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_latest(ckpt_dir: str, model: DCortexV2Model, optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[int, int, List, List, float]:
    paths = list(Path(ckpt_dir).glob("ckpt_campaign_step_*.pt"))
    if not paths:
        return 0, 0, [], [], float("inf")
    paths.sort(key=lambda p: int(p.name.split("_step_")[1].split(".")[0]))
    for path in reversed(paths):
        try:
            c = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(c["model"])
            optimizer.load_state_dict(c["optimizer"])
            rng = c.get("rng")
            if rng is not None:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
                torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in rng["cuda"]])
            print(f"[INFO] Resumed from {path.name} (step {c['step']})", flush=True)
            return (int(c["step"]), int(c.get("epoch", 0)), list(c.get("loss_history", [])),
                    list(c.get("eval_history", [])), float(c.get("best_ppl", float("inf"))))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Corrupt checkpoint {path.name} skipped: {exc}", flush=True)
    return 0, 0, [], [], float("inf")


# ---------------------------------------------------------------------------
# PHASE 2: VRAM autotune
# ---------------------------------------------------------------------------

def autotune(model: DCortexV2Model, train_data: np.memmap, device: torch.device,
             dtype: torch.dtype, vocab: int, vram_limit: float,
             candidates: List[Tuple[int, int]]) -> Dict:
    print(SEP, flush=True)
    print(f"[INFO] PHASE 2 autotune: probing {len(candidates)} (batch, context) "
          f"configs under {vram_limit} GB", flush=True)
    print(SEP, flush=True)
    optimizer = build_lm_optimizer(model, lr=1e-4, weight_decay=0.1)
    results: List[Dict] = []
    best: Optional[Dict] = None
    for batch, context in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok = True
        peak = 0.0
        t0 = time.time()
        n_probe = 5
        try:
            for _ in range(n_probe):
                xt, yt = get_batch(train_data, batch, context, device)
                optimizer.zero_grad(set_to_none=True)
                with silent_stdout():
                    model.reset_memory()
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model.decode(xt)
                    loss = F.cross_entropy(logits.view(-1, vocab), yt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                ok = False
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise
        dt = time.time() - t0
        tok_s = (batch * context * n_probe) / dt if ok else 0.0
        fits = ok and peak < vram_limit
        rec = {"batch": batch, "context": context, "peak_vram_gb": round(peak, 3),
               "tok_s": round(tok_s, 0), "ok": ok, "fits": fits,
               "tokens_per_microstep": batch * context}
        results.append(rec)
        status = "OK" if fits else ("OOM" if not ok else "OVER")
        print(f"  batch={batch:3d} ctx={context:5d} -> peak={peak:5.2f}GB tok/s={tok_s:8.0f} "
              f"[{status}]", flush=True)
        if fits and (best is None or rec["tokens_per_microstep"] > best["tokens_per_microstep"]):
            best = rec
        # free grads/state between probes
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        # Candidates are passed ascending in tokens/microstep: once one OOMs or
        # exceeds the ceiling, every larger config will too, so stop escalating.
        if best is not None and (not ok or peak >= vram_limit):
            print(f"  [INFO] config exceeds ceiling; stopping escalation.", flush=True)
            break
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("Autotune found no config under the VRAM ceiling.")
    print(SEP, flush=True)
    print(f"[INFO] Autotune winner: batch={best['batch']} context={best['context']} "
          f"peak={best['peak_vram_gb']} GB tok/s={best['tok_s']}", flush=True)
    print(SEP, flush=True)
    return {"winner": best, "candidates": results, "vram_limit": vram_limit}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class CampaignConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0


def run(args: argparse.Namespace) -> int:
    device, dtype = setup_device()
    seed_everything(SEED)
    enable_gradient_checkpointing()

    project_root = Path(args.run_dir)
    ckpt_dir = project_root / "checkpoints"
    results_dir = project_root / "results"
    bin_dir = project_root / "dataset_cache" / "bin"
    configs_dir = project_root / "configs"
    for d in [ckpt_dir, results_dir, bin_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Project root: {project_root}", flush=True)

    model_cfg = big_config()
    vocab = model_cfg.vocab_size

    train_bin, val_bin = tokenize_corpus(args.lm_source, str(bin_dir),
                                          args.train_tokens, args.val_tokens)
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    print(f"[INFO] Data: {len(train_data):,} train / {len(val_data):,} val tokens", flush=True)

    model = DCortexV2Model(model_cfg).to(device)

    # ---- PHASE 2: autotune ----
    if args.phase in ("autotune", "all"):
        cands = [(int(b), int(c)) for b, c in (pair.split(":") for pair in args.candidates.split(","))]
        at = autotune(model, train_data, device, dtype, vocab, args.vram_limit, cands)
        win = at["winner"]
        tokens_per_micro = win["batch"] * win["context"]
        grad_accum = max(1, round(args.target_eff_tokens / tokens_per_micro))
        tokens_per_step = tokens_per_micro * grad_accum
        steps_per_sec = win["tok_s"] / max(1, tokens_per_step)
        suggested_steps = int(steps_per_sec * args.target_minutes * 60)
        suggested_steps = max(args.min_steps, min(args.max_steps, suggested_steps))
        at["grad_accum"] = grad_accum
        at["tokens_per_step"] = tokens_per_step
        at["suggested_total_steps"] = suggested_steps
        at["target_minutes"] = args.target_minutes
        with open(results_dir / "autotune.json", "w", encoding="utf-8") as handle:
            json.dump(at, handle, indent=2)
        print(f"[INFO] grad_accum={grad_accum} -> tokens/step={tokens_per_step:,} | "
              f"suggested_total_steps={suggested_steps} (target {args.target_minutes} min)",
              flush=True)
        print("AUTOTUNE_JSON " + json.dumps({"winner": win, "grad_accum": grad_accum,
              "tokens_per_step": tokens_per_step, "suggested_total_steps": suggested_steps}),
              flush=True)
        if args.phase == "autotune":
            return 0

    # ---- PHASE 3: training ----
    at_path = results_dir / "autotune.json"
    if args.batch > 0 and args.context > 0:
        batch, context, grad_accum = args.batch, args.context, args.grad_accum
    elif at_path.exists():
        at = json.loads(at_path.read_text(encoding="utf-8"))
        batch = at["winner"]["batch"]
        context = at["winner"]["context"]
        grad_accum = args.grad_accum if args.grad_accum > 0 else at["grad_accum"]
    else:
        raise RuntimeError("No (batch, context) given and no autotune.json present.")

    total_steps = args.total_steps
    if total_steps <= 0:
        at = json.loads(at_path.read_text(encoding="utf-8")) if at_path.exists() else {}
        total_steps = at.get("suggested_total_steps", 1000)
    warmup = max(10, min(200, total_steps // 10))
    eval_every = args.eval_every if args.eval_every > 0 else max(50, total_steps // 12)
    cc = CampaignConfig()

    optimizer = build_lm_optimizer(model, lr=cc.lr, weight_decay=cc.weight_decay)
    start_step, epoch, loss_history, eval_history, best_ppl = 0, 0, [], [], float("inf")
    if not args.fresh:
        start_step, epoch, loss_history, eval_history, best_ppl = load_latest(
            str(ckpt_dir), model, optimizer, device)

    eval_win_periodic = eval_windows(val_data, args.eval_windows_periodic, EVAL_CONTEXT)
    eval_win_full = eval_windows(val_data, args.eval_windows_full, EVAL_CONTEXT)

    # Step-0 baseline (untrained big model) on the held-out windows.
    baseline_ce, baseline_ppl = (None, None)
    if start_step == 0:
        baseline_ce, baseline_ppl = eval_perplexity(model, val_data, eval_win_full,
                                                     EVAL_CONTEXT, device, dtype, vocab)
        print(f"[INFO] Step-0 untrained baseline held-out ppl = {baseline_ppl:.2f} "
              f"(CE {baseline_ce:.4f})", flush=True)

    print(SEP, flush=True)
    print(f"[INFO] PHASE 3 LM_DECODER campaign | batch={batch} context={context} "
          f"grad_accum={grad_accum} | total_steps={total_steps} warmup={warmup} "
          f"eval_every={eval_every}", flush=True)
    print(f"[INFO] tokens/step={batch * context * grad_accum:,} | source={os.path.basename(args.lm_source)}",
          flush=True)
    print(SEP, flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    patience_left = args.patience
    overfit_divergence = False
    best_step = start_step
    stop_reason = "max_steps"

    for step in range(start_step, total_steps):
        lr = cosine_lr(step, cc.lr, cc.min_lr, warmup, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_t0 = time.time()
        step_loss = 0.0
        try:
            for _ in range(grad_accum):
                xt, yt = get_batch(train_data, batch, context, device)
                with silent_stdout():
                    model.reset_memory()
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model.decode(xt)
                    loss = F.cross_entropy(logits.view(-1, vocab), yt.view(-1))
                    scaled = loss / grad_accum
                scaled.backward()
                step_loss += loss.item() / grad_accum
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] OOM at step {step}, recovering and skipping.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cc.grad_clip).item()
        optimizer.step()
        loss_history.append((step, step_loss))

        if step % args.log_every == 0 or step == start_step or step == total_steps - 1:
            dt = max(1e-6, time.time() - step_t0)
            tok_s = batch * context * grad_accum / dt
            elapsed = time.time() - t_start
            done = step - start_step + 1
            eta_s = elapsed / max(1, done) * (total_steps - step - 1)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Step {step:5d}/{total_steps} | train_loss {step_loss:.4f} | "
                  f"tok/s {tok_s:8.0f} | gn {grad_norm:.2f} | peak {peak:.2f}GB | "
                  f"ETA {int(eta_s // 60)}m{int(eta_s % 60):02d}s", flush=True)

        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            v_ce, v_ppl = eval_perplexity(model, val_data, eval_win_periodic,
                                          EVAL_CONTEXT, device, dtype, vocab)
            eval_history.append((step + 1, v_ce, v_ppl))
            recent_train = np.mean([l for _s, l in loss_history[-max(1, eval_every // args.log_every):]])
            improved = v_ppl < best_ppl - 1e-6
            print(f"  [EVAL] step={step + 1} | held-out ppl={v_ppl:.2f} (CE {v_ce:.4f}) | "
                  f"best={best_ppl if best_ppl != float('inf') else float('nan'):.2f} | "
                  f"train_loss~{recent_train:.4f} | {'IMPROVED' if improved else 'no-improve'}",
                  flush=True)
            if improved:
                best_ppl = v_ppl
                best_step = step + 1
                patience_left = args.patience
                save_ckpt(str(results_dir / "best_model.pt"), model, optimizer, step + 1,
                          epoch, loss_history, eval_history, best_ppl, model_cfg,
                          {"route": "LM_DECODER", "batch": batch, "context": context,
                           "grad_accum": grad_accum, "best_step": best_step,
                           "baseline_ppl": baseline_ppl, "label": LABEL})
                print(f"  [INFO] best_model.pt saved (ppl {best_ppl:.2f} @ step {best_step})",
                      flush=True)
            else:
                patience_left -= 1
                if recent_train < np.mean([l for _s, l in loss_history[:max(1, eval_every // args.log_every)]]):
                    # train still improving but val not -> divergence signal
                    if v_ppl > best_ppl * 1.02:
                        overfit_divergence = True
                if patience_left <= 0:
                    stop_reason = "early_stopping (val plateau/divergence)"
                    print(f"  [WARN] Early stopping at step {step + 1} "
                          f"(no held-out improvement, patience exhausted).", flush=True)
                    save_ckpt(str(ckpt_dir / f"ckpt_campaign_step_{step + 1:06d}.pt"), model,
                              optimizer, step + 1, epoch, loss_history, eval_history,
                              best_ppl, model_cfg, {"route": "LM_DECODER"})
                    break

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(str(ckpt_dir / f"ckpt_campaign_step_{step + 1:06d}.pt"), model,
                      optimizer, step + 1, epoch, loss_history, eval_history, best_ppl,
                      model_cfg, {"route": "LM_DECODER"})

        if math.isnan(step_loss) or math.isinf(step_loss):
            stop_reason = "non_finite_loss"
            print(f"[ERROR] Non-finite loss at step {step}.", flush=True)
            break

        if args.max_minutes > 0 and (time.time() - t_start) / 60.0 > args.max_minutes:
            stop_reason = "wall_clock"
            print(f"[INFO] Wall-clock budget {args.max_minutes} min reached at step "
                  f"{step + 1}. Stopping.", flush=True)
            save_ckpt(str(ckpt_dir / f"ckpt_campaign_step_{step + 1:06d}.pt"), model,
                      optimizer, step + 1, epoch, loss_history, eval_history, best_ppl,
                      model_cfg, {"route": "LM_DECODER"})
            break

    total_time = time.time() - t_start
    final_step = loss_history[-1][0] + 1 if loss_history else start_step
    save_ckpt(str(ckpt_dir / f"ckpt_campaign_step_{final_step:06d}.pt"), model, optimizer,
              final_step, epoch, loss_history, eval_history, best_ppl, model_cfg,
              {"route": "LM_DECODER"})
    peak_overall = torch.cuda.max_memory_allocated() / (1024 ** 3)

    with open(results_dir / "loss_history.json", "w", encoding="utf-8") as handle:
        json.dump({"loss_history": loss_history, "eval_history": eval_history}, handle, indent=2)

    meta = {
        "route": "LM_DECODER", "label": LABEL,
        "source": os.path.basename(args.lm_source),
        "model_params_total_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "trained_params_m": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
        "batch": batch, "context": context, "grad_accum": grad_accum,
        "tokens_per_step": batch * context * grad_accum,
        "total_steps_run": final_step, "total_steps_target": total_steps,
        "warmup": warmup, "lr": cc.lr, "min_lr": cc.min_lr,
        "peak_vram_gb": round(peak_overall, 3), "minutes": round(total_time / 60, 2),
        "baseline_ce": baseline_ce, "baseline_ppl": baseline_ppl,
        "best_ppl": best_ppl if best_ppl != float("inf") else None,
        "best_step": best_step, "stop_reason": stop_reason,
        "overfit_divergence": overfit_divergence,
        "eval_context": EVAL_CONTEXT, "eval_seed": EVAL_SEED,
        "eval_windows_full": args.eval_windows_full,
        "train_tokens": int(len(train_data)), "val_tokens": int(len(val_data)),
        "dtype": str(dtype), "gradient_checkpointing": True,
    }
    with open(results_dir / "campaign_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(SEP, flush=True)
    print(f"[INFO] Campaign done in {total_time / 60:.2f} min | peak {peak_overall:.2f} GB | "
          f"baseline ppl {baseline_ppl} -> best ppl {best_ppl:.2f} @ step {best_step} | "
          f"stop={stop_reason}", flush=True)
    print("CAMPAIGN_META_JSON " + json.dumps(meta), flush=True)
    print(SEP, flush=True)
    return 0


LABEL: str = ("Deliberately undertrained DECODER-BACKBONE pass (LM_DECODER route). "
              "Encoder (Agent A) and memory banks are UNTRAINED (frozen, never in the "
              "forward graph). This is a language-model backbone campaign, not a "
              "memory-recall model.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D_Cortex LM_DECODER training campaign")
    p.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "runs" / "campaign"))
    p.add_argument("--lm-source", type=str, default=DEFAULT_LM_SOURCE)
    p.add_argument("--train-tokens", type=int, default=120_000_000)
    p.add_argument("--val-tokens", type=int, default=8_000_000)
    p.add_argument("--phase", type=str, default="all", choices=["autotune", "train", "all"])
    p.add_argument("--candidates", type=str,
                   default="8:1024,16:1024,24:1024,32:1024,12:2048,20:2048,8:2048")
    p.add_argument("--vram-limit", type=float, default=14.0)
    p.add_argument("--target-eff-tokens", type=int, default=262_144)
    p.add_argument("--target-minutes", type=float, default=60.0)
    p.add_argument("--min-steps", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-windows-periodic", type=int, default=96)
    p.add_argument("--eval-windows-full", type=int, default=512)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-minutes", type=float, default=0.0, help="Wall-clock safety stop (0=off)")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--tag", type=str, default="campaign")
    return p


def main() -> int:
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
