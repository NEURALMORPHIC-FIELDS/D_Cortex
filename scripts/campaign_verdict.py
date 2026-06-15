# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- PHASE 4 quality eval and verdict for the LM_DECODER
# training campaign. Loads best_model.pt, computes held-out perplexity on the
# fixed held-out windows, evaluates a pretrained GPT-2 (124M) reference on the
# IDENTICAL windows, produces qualitative generations, evaluates three
# pre-declared gates, and writes verdict.json plus loss/eval curves. Reads the
# campaign artifacts only (separation of generation and verification).

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch
import torch.nn.functional as F
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from train_campaign import (big_config, eval_windows, eval_perplexity,
                            EVAL_CONTEXT, silent_stdout)

SEP: str = "=" * 70
ENC = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def gpt2_perplexity(model, val_data: np.memmap, windows: List[int], context: int,
                    device: torch.device, dtype: torch.dtype,
                    eval_batch: int = 8) -> Tuple[float, float]:
    """GPT-2 held-out perplexity using the SAME (x, y) windows as our model."""
    vocab = model.config.vocab_size
    total_ce = 0.0
    total_tok = 0
    for b0 in range(0, len(windows), eval_batch):
        idx = windows[b0:b0 + eval_batch]
        x = np.stack([val_data[i:i + context].astype(np.int64) for i in idx])
        y = np.stack([val_data[i + 1:i + 1 + context].astype(np.int64) for i in idx])
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(xt).logits
        ce = F.cross_entropy(logits.view(-1, vocab).float(), yt.view(-1), reduction="sum")
        total_ce += ce.item()
        total_tok += yt.numel()
    mean_ce = total_ce / max(1, total_tok)
    return mean_ce, math.exp(mean_ce)


@torch.no_grad()
def generate(model: DCortexV2Model, prompt: str, max_new: int, device: torch.device,
             dtype: torch.dtype, temperature: float = 0.8, top_k: int = 40) -> str:
    """Autoregressive sampling from the decoder (empty memory, read-only)."""
    ids = ENC.encode_ordinary(prompt)
    with silent_stdout():
        model.reset_memory()
    for _ in range(max_new):
        ctx = ids[-1024:]
        xt = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model.decode(xt)
        next_logits = logits[0, -1].float() / max(1e-6, temperature)
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
        if nxt == ENC.eot_token:
            break
    return ENC.decode(ids)


def plot_curves(loss_history: List, eval_history: List, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if loss_history:
        steps = [s for s, _ in loss_history]
        losses = [l for _, l in loss_history]
        ax1.plot(steps, losses, color="#1f77b4", linewidth=1.0, label="train loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("train loss (CE)", color="#1f77b4")
    if eval_history:
        ax2 = ax1.twinx()
        es = [e[0] for e in eval_history]
        ppl = [e[2] for e in eval_history]
        ax2.plot(es, ppl, color="#d62728", marker="o", linewidth=1.2, label="held-out ppl")
        ax2.set_ylabel("held-out perplexity", color="#d62728")
    plt.title("D_Cortex LM_DECODER campaign (deliberately undertrained backbone)")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[INFO] Curves saved: {out_path}", flush=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="PHASE 4 campaign verdict")
    parser.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "runs" / "campaign"))
    parser.add_argument("--gpt2", type=str, default="gpt2")  # gpt2 = 124M
    parser.add_argument("--gen-new", type=int, default=120)
    parser.add_argument("--drop-target", type=float, default=0.30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_dir = run_dir / "results"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    meta = json.loads((results_dir / "campaign_meta.json").read_text(encoding="utf-8"))
    history = json.loads((results_dir / "loss_history.json").read_text(encoding="utf-8"))
    baseline_ppl = meta["baseline_ppl"]
    label = meta["label"]
    vocab = DCortexConfig().vocab_size

    val_data = np.memmap(run_dir / "dataset_cache" / "bin" / "campaign_val.bin",
                         dtype=np.uint16, mode="r")
    windows = eval_windows(val_data, meta["eval_windows_full"], EVAL_CONTEXT)

    print(SEP, flush=True)
    print("[INFO] PHASE 4: loading best_model.pt", flush=True)
    ckpt = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
    model = DCortexV2Model(big_config()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    reloadable = True
    best_step = ckpt.get("step")
    print(f"[INFO] best_model.pt reloaded (step {best_step}, recorded ppl "
          f"{ckpt.get('best_ppl'):.2f})", flush=True)

    our_ce, our_ppl = eval_perplexity(model, val_data, windows, EVAL_CONTEXT,
                                      device, dtype, vocab)
    print(f"[INFO] OUR held-out ppl = {our_ppl:.2f} (CE {our_ce:.4f}) "
          f"over {len(windows)} windows x {EVAL_CONTEXT} ctx", flush=True)

    print(f"[INFO] Loading GPT-2 reference: {args.gpt2}", flush=True)
    from transformers import AutoModelForCausalLM
    gpt2 = AutoModelForCausalLM.from_pretrained(args.gpt2).to(device).eval()
    gpt2_params = sum(p.numel() for p in gpt2.parameters())
    gpt2_ce, gpt2_ppl = gpt2_perplexity(gpt2, val_data, windows, EVAL_CONTEXT, device, dtype)
    print(f"[INFO] GPT-2 ({gpt2_params/1e6:.0f}M) held-out ppl = {gpt2_ppl:.2f} "
          f"(CE {gpt2_ce:.4f}) on IDENTICAL windows", flush=True)
    del gpt2
    torch.cuda.empty_cache()

    # Qualitative generations.
    prompts = [
        "[SYSTEM] You are a knowledgeable AI that provides accurate, well-structured information.\n\n[USER] Summarize this content.\n\n[ASSISTANT]",
        "The history of the Roman Empire",
        "In machine learning, a transformer is",
        "Once upon a time, in a small village,",
    ]
    gens: List[str] = []
    for p in prompts:
        text = generate(model, p, args.gen_new, device, dtype)
        gens.append(text)
    gen_path = results_dir / "generations.txt"
    with open(gen_path, "w", encoding="utf-8") as handle:
        handle.write(label + "\n" + SEP + "\n\n")
        for i, (p, g) in enumerate(zip(prompts, gens)):
            handle.write(f"--- generation {i + 1} (prompt: {p[:60]!r}) ---\n{g}\n\n")
    print(f"[INFO] {len(gens)} generations saved: {gen_path}", flush=True)

    plot_curves(history.get("loss_history", []), history.get("eval_history", []),
                results_dir / "campaign_curves.png")

    drop_pct = (baseline_ppl - our_ppl) / baseline_ppl * 100.0 if baseline_ppl else 0.0
    gate_a = drop_pct >= args.drop_target * 100.0
    gate_b = not bool(meta.get("overfit_divergence"))
    gate_c = reloadable and our_ppl == our_ppl  # finite check

    verdict: List[Dict[str, Any]] = [
        {"criterion_id": "PPL_IMPROVE_30PCT", "passed": bool(gate_a),
         "evidence": (f"held-out ppl {baseline_ppl:.2f} (step-0 untrained) -> {our_ppl:.2f} "
                      f"(best_model); improvement {drop_pct:.1f}% (target >= "
                      f"{args.drop_target*100:.0f}%). GPT-2(124M) reference on identical "
                      f"windows = {gpt2_ppl:.2f}. ABSOLUTE: ours={our_ppl:.2f} ce={our_ce:.4f}, "
                      f"gpt2={gpt2_ppl:.2f} ce={gpt2_ce:.4f}, baseline={baseline_ppl:.2f}.")},
        {"criterion_id": "NO_OVERFIT_DIVERGENCE", "passed": bool(gate_b),
         "evidence": (f"overfit_divergence={meta.get('overfit_divergence')}; "
                      f"stop_reason={meta.get('stop_reason')}; "
                      f"{len(history.get('eval_history', []))} held-out evals recorded.")},
        {"criterion_id": "BEST_MODEL_RELOADABLE", "passed": bool(gate_c),
         "evidence": (f"best_model.pt reloaded (step {best_step}); recomputed held-out "
                      f"ppl {our_ppl:.2f}; {len(gens)} generations produced.")},
    ]
    reference = {
        "label": label,
        "absolute_held_out_ppl": {"baseline_untrained": baseline_ppl,
                                  "ours_best_model": round(our_ppl, 3),
                                  "gpt2_124m_reference": round(gpt2_ppl, 3)},
        "ppl_improvement_pct_vs_baseline": round(drop_pct, 2),
        "gpt2_params_m": round(gpt2_params / 1e6, 1),
        "eval_windows": len(windows), "eval_context": EVAL_CONTEXT,
        "trained_params_m": meta.get("trained_params_m"),
        "model_params_total_m": meta.get("model_params_total_m"),
        "best_step": best_step, "peak_vram_gb": meta.get("peak_vram_gb"),
        "batch": meta.get("batch"), "context": meta.get("context"),
        "grad_accum": meta.get("grad_accum"), "minutes": meta.get("minutes"),
    }
    out = {"verdict": verdict, "reference": reference}
    with open(results_dir / "verdict.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(SEP, flush=True)
    print(f"[INFO] {label}", flush=True)
    print(SEP, flush=True)
    for item in verdict:
        tag = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{tag}  [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] ABSOLUTE held-out ppl: baseline {baseline_ppl:.2f} | "
          f"ours {our_ppl:.2f} | GPT-2(124M) {gpt2_ppl:.2f}", flush=True)
    all_pass = all(i["passed"] for i in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print("VERDICT_JSON " + json.dumps(out), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
