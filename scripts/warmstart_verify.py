# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- PHASE C verification of the gpt2-medium warm-start.
# Uses the SAME held-out split and eval windows as the backbone campaign
# (campaign_val.bin, seed-1234 windows, 512 x 1024). Evaluates three gates:
# MAPPED COUNT, CLEAN WARM-START PPL (vs random baseline and the gpt2-medium
# reference floor), and INERT FRESH COMPONENTS. Emits warmstart_verdict.json.

import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from train_campaign import big_config, eval_windows, eval_perplexity, EVAL_CONTEXT
from campaign_verdict import gpt2_perplexity

SEP: str = "=" * 70
RANDOM_BASELINE_PPL: float = 59193.72445138559   # step-0 untrained, backbone campaign
PPL_GATE: float = 80.0
IMPROVE_FACTOR: float = 100.0


@contextlib.contextmanager
def silent_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def inert_check(model: DCortexV2Model, val_data: np.memmap, device: torch.device,
                vocab: int) -> Dict[str, Any]:
    """With fusion cross-attn zeroed, populating memory must not change logits."""
    x = np.stack([val_data[0:EVAL_CONTEXT].astype(np.int64),
                  val_data[5000:5000 + EVAL_CONTEXT].astype(np.int64)])
    xt = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        with silent_stdout():
            model.reset_memory()
        logits_empty = model.decode(xt).float()
        # Populate memory with a few real facts written to the working bank.
        with silent_stdout():
            model.reset_memory()
        for start in (1000, 2000, 3000):
            fact = torch.from_numpy(
                val_data[start:start + 64].astype(np.int64)).unsqueeze(0).to(device)
            ans = torch.tensor([int(val_data[start + 64])], device=device)
            with silent_stdout():
                model.encode(fact, answer_token_id=ans, lexical_alpha=0.9, force_bank="working")
        logits_full = model.decode(xt).float()
    max_abs_diff = float((logits_empty - logits_full).abs().max().item())
    return {"max_abs_logit_diff": max_abs_diff,
            "inert": max_abs_diff < 1e-3}


def main() -> int:
    run_dir = REPO_ROOT / "runs" / "warmstart"
    ckpt_path = run_dir / "warmstarted_init.pt"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    vocab = DCortexConfig().vocab_size

    val_bin = REPO_ROOT / "runs" / "campaign" / "dataset_cache" / "bin" / "campaign_val.bin"
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    windows = eval_windows(val_data, 512, EVAL_CONTEXT)

    print(SEP, flush=True)
    print("[INFO] PHASE C: loading warmstarted_init.pt", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    model = DCortexV2Model(big_config()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Gate 1: MAPPED COUNT ---
    g1_pass = (manifest["n_standard_blocks_mapped"] == 12
               and manifest["n_fusion_blocks_mapped"] == 4
               and manifest["token_emb_mapped"]
               and manifest["final_norm_mapped"]
               and manifest["logits_sanity"].get("finite", False))
    g1_ev = (f"mapped {manifest['n_mapped_ops']} tensor ops, zeroed "
             f"{manifest['n_zeroed_ops']}; 12 std + 4 fusion blocks + token_emb + "
             f"final_norm mapped; 0 shape errors (build asserted); lm_head_tied="
             f"{manifest['lm_head_tied']}; forward finite="
             f"{manifest['logits_sanity'].get('finite')}.")
    print(f"[INFO] Gate1 MAPPED_COUNT: {g1_ev}", flush=True)

    # --- Gate 2: CLEAN WARM-START PPL ---
    our_ce, our_ppl = eval_perplexity(model, val_data, windows, EVAL_CONTEXT,
                                      device, dtype, vocab)
    print(f"[INFO] warm-start held-out ppl = {our_ppl:.2f} (CE {our_ce:.4f})", flush=True)
    print("[HF] Loading gpt2-medium for reference floor ...", flush=True)
    from transformers import GPT2LMHeadModel
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device).eval()
    gpt2_ce, gpt2_ppl = gpt2_perplexity(gpt2, val_data, windows, EVAL_CONTEXT, device, dtype)
    print(f"[HF] gpt2-medium held-out ppl = {gpt2_ppl:.2f} (CE {gpt2_ce:.4f}) [reference floor]",
          flush=True)
    del gpt2
    torch.cuda.empty_cache()
    improve = RANDOM_BASELINE_PPL / our_ppl
    g2_pass = (our_ppl < PPL_GATE) and (improve > IMPROVE_FACTOR)
    g2_ev = (f"warm-start ppl {our_ppl:.2f} (target < {PPL_GATE:.0f}); improvement "
             f"{improve:.0f}x over random baseline {RANDOM_BASELINE_PPL:.0f} "
             f"(target > {IMPROVE_FACTOR:.0f}x). gpt2-medium reference floor = {gpt2_ppl:.2f}. "
             f"ABSOLUTE: warmstart={our_ppl:.2f} ce={our_ce:.4f}, "
             f"gpt2-medium={gpt2_ppl:.2f} ce={gpt2_ce:.4f}, random={RANDOM_BASELINE_PPL:.0f}.")
    print(f"[INFO] Gate2 CLEAN_WARMSTART_PPL: {g2_ev}", flush=True)

    # --- Gate 3: INERT FRESH COMPONENTS ---
    inert = inert_check(model, val_data, device, vocab)
    g3_pass = bool(inert["inert"])
    g3_ev = (f"populating memory changes logits by max|diff|="
             f"{inert['max_abs_logit_diff']:.2e} (fusion cross_attn.out zeroed); "
             f"inert={g3_pass} (tol 1e-3).")
    print(f"[INFO] Gate3 INERT_FRESH: {g3_ev}", flush=True)

    verdict: List[Dict[str, Any]] = [
        {"criterion_id": "MAPPED_COUNT", "passed": bool(g1_pass), "evidence": g1_ev},
        {"criterion_id": "CLEAN_WARMSTART_PPL", "passed": bool(g2_pass), "evidence": g2_ev},
        {"criterion_id": "INERT_FRESH_COMPONENTS", "passed": bool(g3_pass), "evidence": g3_ev},
    ]
    reference = {
        "absolute_held_out_ppl": {"random_baseline": RANDOM_BASELINE_PPL,
                                  "warmstart": round(our_ppl, 3),
                                  "gpt2_medium_floor": round(gpt2_ppl, 3)},
        "improvement_x_over_random": round(improve, 1),
        "gpt2_layer_indices": manifest["gpt2_layer_indices"],
        "activation_drift": manifest["activation_drift"],
        "eval_windows": len(windows), "eval_context": EVAL_CONTEXT,
        "logits_sanity": manifest["logits_sanity"],
    }
    out = {"verdict": verdict, "reference": reference}
    with open(run_dir / "warmstart_verdict.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(SEP, flush=True)
    for item in verdict:
        tag = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{tag}  [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] ABSOLUTE held-out ppl: random {RANDOM_BASELINE_PPL:.0f} | "
          f"warmstart {our_ppl:.2f} | gpt2-medium floor {gpt2_ppl:.2f}", flush=True)
    all_pass = all(i["passed"] for i in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print("WARMSTART_VERDICT_JSON " + json.dumps(out), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
