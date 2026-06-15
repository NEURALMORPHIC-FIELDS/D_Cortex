# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- full MEMORY training campaign from the gpt2-warm-started
# decoder. Trains the full graph (encoder Agent A + memory banks + decoder)
# jointly on the adapter's episodes. Warm-started decoder weights fine-tune at a
# LOW lr; fresh components (encoder, memory, fusion cross-attn, mem_gate) learn at
# a HIGH lr. The dcortex/ architecture is NOT modified.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "colab", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import StandardTransformerBlock
from train_campaign import big_config, eval_windows, eval_perplexity, EVAL_CONTEXT
from train_memory_probe import (structural_forward, eval_answer_ce, load_episodes,
                                _pad, SEQ_LEN, LEXICAL_ALPHA, ENC, EOT, silent)

SEP: str = "=" * 70
SEED: int = 42
WARMSTART_PPL: float = 46.95


_GC = {"on": False}
_ORIG = StandardTransformerBlock.forward


def _gc_forward(self, h, attention_mask=None):
    if _GC["on"] and self.training and torch.is_grad_enabled() and h.requires_grad:
        return torch_checkpoint.checkpoint(_ORIG, self, h, attention_mask, use_reentrant=False)
    return _ORIG(self, h, attention_mask)


def is_low_group(name: str) -> bool:
    """Warm-started (LOW lr) tensors, by module identity (see warmstart manifest):
    standard decoder blocks, fusion self-attn/FFN/norms, token+pos embeddings,
    dec_emb_norm, final norm. Everything else (encoder, memory, fusion cross-attn,
    mem_gate, readers, addressing, aux head) is fresh -> HIGH lr."""
    if name.startswith("dec_standard_blocks."):
        return True
    if name.startswith("dec_fusion_blocks.") and any(
            k in name for k in (".norm_self", ".self_attn.", ".norm_ff", ".ff.")):
        return True
    if name.startswith(("shared_token_emb", "shared_pos_emb", "dec_emb_norm",
                        "dec_final_norm")):
        return True
    return False


def partition_params(model: DCortexV2Model, manifest: Dict
                     ) -> Tuple[List, List, Dict[str, Any]]:
    low, high = [], []
    low_n, high_n = 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_low_group(name):
            low.append(p)
            low_n += p.numel()
        else:
            high.append(p)
            high_n += p.numel()
    # Cross-check against the manifest: fresh keys should land in HIGH.
    fresh = set(manifest.get("fresh_state_dict_keys", []))
    misplaced = 0
    for name, _p in model.named_parameters():
        if name in fresh and is_low_group(name):
            misplaced += 1
    info = {"low_params_m": round(low_n / 1e6, 2), "high_params_m": round(high_n / 1e6, 2),
            "low_count": len(low), "high_count": len(high),
            "manifest_fresh_misplaced_in_low": misplaced}
    return low, high, info


def load_episodes_cap(path: str, cap: int) -> List[Dict]:
    eps: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
            if len(eps) >= cap:
                break
    return eps


def group_lr(step: int, peak: float, warmup: int, total: int) -> float:
    min_lr = 0.1 * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * t))


def save_ckpt(path: str, model, opt, step, epoch, loss_hist, eval_hist, best_metric,
              model_cfg, meta) -> None:
    ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
            "epoch": epoch, "loss_history": loss_hist, "eval_history": eval_hist,
            "best_metric": best_metric, "config_model": asdict_cfg(model_cfg), "meta": meta,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def asdict_cfg(cfg: DCortexConfig) -> Dict:
    return {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}


def rebundle_for_k(episodes: List[Dict], k: int, n_eps: int, seed: int) -> List[Dict]:
    """Build n_eps episodes of K distinct-subject facts from held-out episode facts
    (for the recall-vs-K characterization)."""
    facts: List[Dict] = []
    seen = set()
    for ep in episodes:
        for f in ep["facts"]:
            if f.get("subject") == "[system]" or not f.get("value"):
                continue
            key = (f["subject"].lower(), f["answer_token_id"])
            if key in seen:
                continue
            seen.add(key)
            facts.append(f)
    rng = random.Random(seed)
    rng.shuffle(facts)
    out: List[Dict] = []
    i = 0
    while i + k <= len(facts) and len(out) < n_eps:
        bundle = facts[i:i + k]
        i += k
        subs = {b["subject"].lower() for b in bundle}
        if len(subs) < k:
            continue
        tgt = rng.randrange(k)
        # query for the target fact: clause truncated before its value token
        f = bundle[tgt]
        ids = ENC.encode_ordinary(f["text"])
        a = int(f["answer_token_id"])
        if a in ids:
            cut = ids.index(a)
            prompt = ENC.decode(ids[:cut]).strip()
        else:
            prompt = f["text"]
        if not prompt:
            continue
        out.append({"facts": bundle, "prompt": prompt, "target_fact_idx": tgt,
                    "answer_token_id": a, "ep_type": "cloze_k"})
    return out


def run(args: argparse.Namespace) -> int:
    print(SEP, flush=True)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[INFO] GPU {name} SM {cap[0]}.{cap[1]} | bf16, NO GradScaler | "
          f"VRAM ceiling {args.vram_limit} GB", flush=True)
    if cap[0] < 8:
        raise RuntimeError("Campaign requires capability >= 8 (bf16).")
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dtype = torch.bfloat16
    StandardTransformerBlock.forward = _gc_forward
    _GC["on"] = True

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    for d in (ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(args.adapter_dir)
    print(f"[INFO] Loading episodes (cap {args.max_train_episodes:,} train)...", flush=True)
    train_eps = load_episodes_cap(str(adapter_dir / "episodes_train.jsonl"),
                                  args.max_train_episodes)
    held_eps = load_episodes_cap(str(adapter_dir / "episodes_heldout.jsonl"),
                                 args.max_heldout_episodes)
    print(f"[INFO] {len(train_eps):,} train / {len(held_eps):,} heldout episodes", flush=True)

    # --- Model + warm-start ---
    model_cfg = big_config()
    model = DCortexV2Model(model_cfg).to(device)
    ws = torch.load(args.warmstart, map_location=device, weights_only=False)
    model.load_state_dict(ws["model"])
    manifest = ws["manifest"]
    print(f"[INFO] Loaded warm-start {os.path.basename(args.warmstart)}", flush=True)

    # NO-FORGET reference: warm-start backbone ppl must reproduce before training.
    backbone_val = np.memmap(REPO_ROOT / "runs" / "campaign" / "dataset_cache" / "bin" /
                             "campaign_val.bin", dtype=np.uint16, mode="r")
    nf_windows = eval_windows(backbone_val, args.noforget_windows, EVAL_CONTEXT)
    nf_full_windows = eval_windows(backbone_val, 512, EVAL_CONTEXT)
    _ce, ws_ppl = eval_perplexity(model, backbone_val, nf_windows, EVAL_CONTEXT, device,
                                  dtype, model_cfg.vocab_size)
    print(f"[INFO] Warm-start backbone ppl reproduces: {ws_ppl:.2f} (ref ~{WARMSTART_PPL})",
          flush=True)
    noforget_threshold = args.noforget_mult * WARMSTART_PPL

    # --- Param-group optimizer ---
    low, high, pinfo = partition_params(model, manifest)
    print(f"[INFO] LOW(lr {args.low_lr}) {pinfo['low_params_m']}M / "
          f"HIGH(lr {args.high_lr}) {pinfo['high_params_m']}M | "
          f"manifest_fresh_misplaced_in_low={pinfo['manifest_fresh_misplaced_in_low']}",
          flush=True)
    optimizer = torch.optim.AdamW([
        {"params": low, "lr": args.low_lr, "weight_decay": 0.1},
        {"params": high, "lr": args.high_lr, "weight_decay": 0.1},
    ], betas=(0.9, 0.95))
    peaks = [args.low_lr, args.high_lr]

    # --- Autotune: confirm one episode + grad_accum fits under the ceiling ---
    print(SEP, flush=True)
    print("[INFO] PHASE 1 autotune: full-graph DUAL_AGENT peak VRAM", flush=True)
    chosen_accum = 0
    autotune_peak = 0.0
    step_seconds = 0.0
    for ga in args.accum_candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok = True
        t0 = time.time()
        try:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                for j in range(ga):
                    ep = train_eps[j % len(train_eps)]
                    with torch.amp.autocast("cuda", dtype=dtype):
                        total, _e, _t = structural_forward(model, ep, device, zero_memory=False)
                        (total / ga).backward()
                    model.clear_overlays()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                ok = False
                gc.collect(); torch.cuda.empty_cache()
            else:
                raise
        dt = (time.time() - t0) / 3.0
        fits = ok and peak < args.vram_limit
        print(f"  grad_accum={ga:3d} -> peak={peak:5.2f}GB  {dt:.2f}s/step  "
              f"[{'OK' if fits else ('OOM' if not ok else 'OVER')}]", flush=True)
        if fits:
            chosen_accum = ga
            autotune_peak = peak
            step_seconds = dt
        if (not ok) or (ok and peak >= args.vram_limit):
            break
    if chosen_accum == 0:
        print("[ERROR] No grad_accum fits under the VRAM ceiling. STOP.", flush=True)
        verdict = [{"criterion_id": "VRAM_FIT", "passed": False,
                    "evidence": f"no config < {args.vram_limit} GB for the full-graph DUAL_AGENT."}]
        with open(results_dir / "verdict.json", "w", encoding="utf-8") as h:
            json.dump({"verdict": verdict}, h, indent=2)
        print("CAMPAIGN_VERDICT_JSON " + json.dumps({"verdict": verdict}), flush=True)
        return 1
    eta_min = step_seconds * args.max_steps / 60.0
    print(f"[INFO] chosen grad_accum={chosen_accum} peak={autotune_peak:.2f}GB "
          f"~{step_seconds:.2f}s/step -> {args.max_steps} steps ~= {eta_min:.0f} min", flush=True)
    print(SEP, flush=True)

    # --- Reset optimizer state allocated during autotune ---
    optimizer.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()

    # --- Pre-training held-out memory metrics (reference) ---
    pre = eval_answer_ce(model, held_eps, device, dtype, args.eval_episodes)
    print(f"[INFO] pre-train held-out answer CE enabled={pre['ce_enabled']:.4f} "
          f"zeroed={pre['ce_zeroed']:.4f}", flush=True)

    print(SEP, flush=True)
    print(f"[INFO] PHASE 2 campaign: {args.max_steps} steps x {chosen_accum} episodes, "
          f"warmup {args.warmup}, eval every {args.eval_every}, patience {args.patience}",
          flush=True)
    print(SEP, flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    loss_hist: List[float] = []
    eval_hist: List[Dict] = []
    order = list(range(len(train_eps)))
    random.shuffle(order)
    ptr = 0
    best_ce = float("inf")
    best_step = 0
    peak_uplift = 0.0
    patience_left = args.patience
    stop_reason = "max_steps"
    forget_step: Optional[int] = None

    for step in range(args.max_steps):
        lr_low = group_lr(step, peaks[0], args.warmup, args.max_steps)
        lr_high = group_lr(step, peaks[1], args.warmup, args.max_steps)
        optimizer.param_groups[0]["lr"] = lr_low
        optimizer.param_groups[1]["lr"] = lr_high
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        toks = 0
        t_step = time.time()
        try:
            for _ in range(chosen_accum):
                if ptr >= len(order):
                    random.shuffle(order); ptr = 0
                ep = train_eps[order[ptr]]; ptr += 1
                with torch.amp.autocast("cuda", dtype=dtype):
                    total, _emit, _t1 = structural_forward(model, ep, device, zero_memory=False)
                    (total / chosen_accum).backward()
                model.clear_overlays()
                step_loss += float(total.item()) / chosen_accum
                toks += (len(ep["facts"]) + 1) * SEQ_LEN
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] OOM at step {step}, recovering.", flush=True)
                optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                continue
            raise
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_hist.append(step_loss)

        if step % args.log_every == 0 or step == args.max_steps - 1:
            dt = time.time() - t_start
            done = step + 1
            eta = dt / done * (args.max_steps - done)
            tok_s = toks / max(1e-6, time.time() - t_step)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Step {step:5d}/{args.max_steps} | train_loss {step_loss:.4f} | "
                  f"tok/s {tok_s:6.0f} | peak {peak:.2f}GB | "
                  f"ETA {int(eta//60)}m{int(eta%60):02d}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step == args.max_steps - 1:
            mem = eval_answer_ce(model, held_eps, device, dtype, args.eval_episodes)
            _nfce, nf_ppl = eval_perplexity(model, backbone_val, nf_windows, EVAL_CONTEXT,
                                            device, dtype, model_cfg.vocab_size)
            uplift = (mem["ce_zeroed"] - mem["ce_enabled"]) / max(1e-9, mem["ce_zeroed"])
            peak_uplift = max(peak_uplift, uplift)
            ev = {"step": step + 1, "ce_enabled": mem["ce_enabled"],
                  "ce_zeroed": mem["ce_zeroed"], "uplift": uplift,
                  "top1_enabled": mem["top1_enabled"], "top1_zeroed": mem["top1_zeroed"],
                  "backbone_ppl": nf_ppl}
            eval_hist.append(ev)
            print(f"  [EVAL] step={step+1} | mem CE enabled={mem['ce_enabled']:.4f} "
                  f"zeroed={mem['ce_zeroed']:.4f} uplift={uplift:.1%} | "
                  f"recall(top1) enabled={mem['top1_enabled']:.1%} zeroed={mem['top1_zeroed']:.1%} | "
                  f"backbone ppl={nf_ppl:.2f} (thr {noforget_threshold:.0f})", flush=True)
            if mem["ce_enabled"] < best_ce - 1e-5:
                best_ce = mem["ce_enabled"]; best_step = step + 1
                patience_left = args.patience
                save_ckpt(str(results_dir / "best_model.pt"), model, optimizer, step + 1,
                          0, loss_hist, eval_hist, best_ce, model_cfg,
                          {"route": "MEMORY", "best_step": best_step, "uplift": uplift,
                           "recall_enabled": mem["top1_enabled"], "backbone_ppl": nf_ppl})
                print(f"  [INFO] best_model.pt saved (held-out CE {best_ce:.4f})", flush=True)
            else:
                patience_left -= 1
            # Early-stop conditions (pre-declared).
            if nf_ppl > noforget_threshold:
                forget_step = step + 1
                stop_reason = f"NO_FORGET breach (backbone ppl {nf_ppl:.1f} > {noforget_threshold:.0f})"
                print(f"  [ERROR] {stop_reason} at step {step+1}. Stopping.", flush=True)
                break
            if patience_left <= 0:
                stop_reason = "early_stop (held-out CE plateau, patience exhausted)"
                print(f"  [WARN] {stop_reason} at step {step+1}.", flush=True)
                break

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(str(ckpt_dir / f"ckpt_memory_step_{step+1:06d}.pt"), model, optimizer,
                      step + 1, 0, loss_hist, eval_hist, best_ce, model_cfg, {"route": "MEMORY"})

    total_time = time.time() - t_start
    final_step = (loss_hist and len(loss_hist)) or 0
    save_ckpt(str(ckpt_dir / f"ckpt_memory_step_{final_step:06d}.pt"), model, optimizer,
              final_step, 0, loss_hist, eval_hist, best_ce, model_cfg, {"route": "MEMORY"})
    peak_overall = torch.cuda.max_memory_allocated() / (1024 ** 3)
    with open(results_dir / "loss_history.json", "w", encoding="utf-8") as h:
        json.dump({"loss_history": loss_hist, "eval_history": eval_hist}, h, indent=2)

    # --- Final evaluation from best_model ---
    print(SEP, flush=True)
    print("[INFO] PHASE 3 final eval from best_model.pt", flush=True)
    best = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"]); model.eval()
    reload_ok = True
    fin = eval_answer_ce(model, held_eps, device, dtype, max(args.eval_episodes, 1000))
    final_uplift = (fin["ce_zeroed"] - fin["ce_enabled"]) / max(1e-9, fin["ce_zeroed"])
    _fce, final_ppl = eval_perplexity(model, backbone_val, nf_full_windows, EVAL_CONTEXT,
                                      device, dtype, model_cfg.vocab_size)
    print(f"[INFO] final held-out: enabled CE={fin['ce_enabled']:.4f} zeroed={fin['ce_zeroed']:.4f} "
          f"uplift={final_uplift:.1%} | recall enabled={fin['top1_enabled']:.1%} "
          f"zeroed={fin['top1_zeroed']:.1%} | backbone ppl(512)={final_ppl:.2f}", flush=True)

    # K-characterization (report, not a gate).
    k_curve = []
    for k in [2, 4, 6, 8, 12, 16]:
        keps = rebundle_for_k(held_eps, k, 400, SEED + k)
        if not keps:
            continue
        km = eval_answer_ce(model, keps, device, dtype, len(keps))
        k_curve.append({"K": k, "n": len(keps), "recall_enabled": km["top1_enabled"],
                        "recall_zeroed": km["top1_zeroed"], "uplift":
                        (km["ce_zeroed"] - km["ce_enabled"]) / max(1e-9, km["ce_zeroed"])})
        print(f"  [K={k:2d}] recall enabled={km['top1_enabled']:.1%} zeroed={km['top1_zeroed']:.1%} "
              f"uplift={k_curve[-1]['uplift']:.1%} (n={len(keps)})", flush=True)

    # --- Pre-declared gates ---
    g_a = final_uplift >= args.uplift_gate and fin["ce_enabled"] < fin["ce_zeroed"]
    g_b = (fin["top1_enabled"] >= args.recall_floor) and (fin["top1_enabled"] > fin["top1_zeroed"])
    g_c = final_ppl < noforget_threshold and forget_step is None
    no_collapse = final_uplift >= 0.5 * peak_uplift
    g_d = no_collapse and (stop_reason != "early_stop (held-out CE plateau, patience exhausted)" or final_uplift >= args.uplift_gate)
    g_e = reload_ok and (fin["ce_enabled"] == fin["ce_enabled"])

    verdict = [
        {"criterion_id": "HELDOUT_MEMORY_UPLIFT", "passed": bool(g_a),
         "evidence": (f"held-out answer CE enabled={fin['ce_enabled']:.4f} vs "
                      f"zeroed={fin['ce_zeroed']:.4f} = {final_uplift:.1%} lower "
                      f"(target >= {args.uplift_gate:.0%}); n={max(args.eval_episodes,1000)} unseen.")},
        {"criterion_id": "RECALL", "passed": bool(g_b),
         "evidence": (f"held-out exact-match recall enabled={fin['top1_enabled']:.1%} "
                      f"(floor {args.recall_floor:.0%}) vs zeroed control "
                      f"{fin['top1_zeroed']:.1%}.")},
        {"criterion_id": "NO_FORGET", "passed": bool(g_c),
         "evidence": (f"backbone LM ppl(512x1024) end={final_ppl:.2f} < {noforget_threshold:.0f} "
                      f"(2x warm-start {WARMSTART_PPL}); breach_step={forget_step}.")},
        {"criterion_id": "NO_BYPASS_NO_OVERFIT", "passed": bool(g_d),
         "evidence": (f"final uplift {final_uplift:.1%} vs peak {peak_uplift:.1%} "
                      f"(no collapse={no_collapse}); stop_reason={stop_reason}.")},
        {"criterion_id": "BEST_MODEL_RELOADABLE", "passed": bool(g_e),
         "evidence": (f"best_model.pt (step {best['step']}) reloaded; recomputed held-out "
                      f"enabled CE={fin['ce_enabled']:.4f}, uplift {final_uplift:.1%}.")},
    ]
    out = {"verdict": verdict, "reference": {
        "warmstart_backbone_ppl_reproduced": round(ws_ppl, 3),
        "final_backbone_ppl": round(final_ppl, 3), "noforget_threshold": round(noforget_threshold, 1),
        "final_heldout_ce_enabled": round(fin["ce_enabled"], 4),
        "final_heldout_ce_zeroed": round(fin["ce_zeroed"], 4),
        "final_uplift": round(final_uplift, 4), "peak_uplift": round(peak_uplift, 4),
        "final_recall_enabled": round(fin["top1_enabled"], 4),
        "final_recall_zeroed": round(fin["top1_zeroed"], 4),
        "best_step": best_step, "steps_run": final_step, "stop_reason": stop_reason,
        "grad_accum": chosen_accum, "autotune_peak_gb": round(autotune_peak, 3),
        "peak_vram_gb": round(peak_overall, 3), "minutes": round(total_time / 60, 2),
        "low_params_m": pinfo["low_params_m"], "high_params_m": pinfo["high_params_m"],
        "k_characterization": k_curve, "eval_history": eval_hist,
        "validation_note": ("single-seed single-env MEASURED ablation, not legal-grade; "
                            ">=20 seeds required to PROVE any external memory-uplift figure.")}}
    with open(results_dir / "verdict.json", "w", encoding="utf-8") as h:
        json.dump(out, h, indent=2)

    print(SEP, flush=True)
    for item in verdict:
        tag = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{tag}  [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(i["passed"] for i in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'} | "
          f"best_model.pt @ step {best_step} | {total_time/60:.1f} min", flush=True)
    print("CAMPAIGN_VERDICT_JSON " + json.dumps(out), flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D_Cortex full memory training campaign")
    p.add_argument("--adapter-dir", type=str, default=str(REPO_ROOT / "runs" / "adapter"))
    p.add_argument("--warmstart", type=str,
                   default=str(REPO_ROOT / "runs" / "warmstart" / "warmstarted_init.pt"))
    p.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "runs" / "memory_campaign"))
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--low-lr", type=float, default=2e-5)
    p.add_argument("--high-lr", type=float, default=2e-4)
    p.add_argument("--vram-limit", type=float, default=14.0)
    p.add_argument("--accum-candidates", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--eval-episodes", type=int, default=800)
    p.add_argument("--noforget-windows", type=int, default=128)
    p.add_argument("--noforget-mult", type=float, default=2.0)
    p.add_argument("--uplift-gate", type=float, default=0.15)
    p.add_argument("--recall-floor", type=float, default=0.05)
    p.add_argument("--max-train-episodes", type=int, default=400000)
    p.add_argument("--max-heldout-episodes", type=int, default=20000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=25)
    return p


def main() -> int:
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
