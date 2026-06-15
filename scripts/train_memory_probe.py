# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- adapter validation: run the v11 DUAL_AGENT structural
# path on the adapter's real episodes (no model change), then measure on HELD-OUT
# episodes whether memory reads carry the answer (ENABLED vs ZEROED answer-token
# cross-entropy). Validates the ROUNDTRIP, PATH_ACCEPTS, and MEMORY_ABLATION
# gates and emits verdict.json (PARSE is taken from the adapter stats).

import argparse
import contextlib
import io
import json
import math
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
SEQ_LEN: int = 64
LEXICAL_ALPHA: float = 0.9
ENC = tiktoken.get_encoding("gpt2")
EOT: int = ENC.eot_token

_GC = {"on": False}
_ORIG = StandardTransformerBlock.forward


def _gc_forward(self, h, attention_mask=None):
    if _GC["on"] and self.training and torch.is_grad_enabled() and h.requires_grad:
        return torch_checkpoint.checkpoint(_ORIG, self, h, attention_mask, use_reentrant=False)
    return _ORIG(self, h, attention_mask)


@contextlib.contextmanager
def silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _pad(ids: List[int], length: int) -> List[int]:
    return ids[:length] if len(ids) > length else ids + [EOT] * (length - len(ids))


def load_episodes(path: str) -> List[Dict]:
    eps: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    return eps


def structural_forward(model: DCortexV2Model, ep: Dict, device: torch.device,
                       zero_memory: bool = False) -> Tuple[torch.Tensor, float, float]:
    """v11 structural episode loss with an optional memory-zero ablation switch.
    Returns (total_loss, answer_ce, top1). Loss composition is preserved."""
    with silent():
        model.reset_memory()
    model.begin_episode()
    fact_keys: List[torch.Tensor] = []
    for fact in ep["facts"]:
        f_ids = _pad(ENC.encode_ordinary(fact["text"]) + [EOT], SEQ_LEN)
        xf = torch.tensor([f_ids], dtype=torch.long, device=device)
        ans = torch.tensor([int(fact["answer_token_id"])], dtype=torch.long, device=device)
        aux = model.encode(xf, answer_token_id=ans, lexical_alpha=LEXICAL_ALPHA,
                           force_bank="working")
        fact_keys.append(aux["w_k_ent"][0])

    p_ids = ENC.encode_ordinary(ep["prompt"])
    xp = torch.tensor([_pad(p_ids, SEQ_LEN)], dtype=torch.long, device=device)
    _, retrieved = model.decode(xp, return_retrieved=True)
    if zero_memory:
        retrieved = torch.zeros_like(retrieved)
    aux_logits = model.aux_answer_head(retrieved)
    target = torch.tensor([int(ep["answer_token_id"])], device=device)
    l_emit = F.cross_entropy(aux_logits, target)

    b, t = xp.shape
    pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    q_emb = model.shared_token_emb(xp) + model.shared_pos_emb(pos)
    q_addr = model.shared_address_encoder(q_emb)
    q_k_ent, _, _ = model.shared_query_engine(q_addr)
    keys = F.normalize(torch.stack(fact_keys, dim=0), dim=-1)
    q_n = F.normalize(q_k_ent, dim=-1)
    sim = (q_n @ keys.t()).squeeze(0)
    l_sel = -F.log_softmax(sim * 5.0, dim=-1)[int(ep["target_fact_idx"])]

    l_sep = torch.tensor(0.0, device=device)
    if len(fact_keys) >= 2:
        sims = keys @ keys.t()
        mask = torch.eye(len(fact_keys), device=device, dtype=torch.bool)
        l_sep = F.relu(sims[~mask] - 0.5).pow(2).mean()

    total = 1.0 * l_emit + 1.0 * l_sel + 0.5 * l_sep
    with torch.no_grad():
        top1 = float(aux_logits[0].argmax().item() == int(ep["answer_token_id"]))
    return total, float(l_emit.item()), top1


@torch.no_grad()
def eval_answer_ce(model: DCortexV2Model, episodes: List[Dict], device: torch.device,
                   dtype: torch.dtype, n: int) -> Dict[str, float]:
    """Held-out answer-token CE with memory reads ENABLED vs ZEROED (one pass)."""
    model.eval()
    ce_on = 0.0
    ce_off = 0.0
    top1_on = 0
    top1_off = 0
    for ep in episodes[:n]:
        with silent():
            model.reset_memory()
        for fact in ep["facts"]:
            f_ids = _pad(ENC.encode_ordinary(fact["text"]) + [EOT], SEQ_LEN)
            xf = torch.tensor([f_ids], dtype=torch.long, device=device)
            ans = torch.tensor([int(fact["answer_token_id"])], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=dtype):
                model.encode(xf, answer_token_id=ans, lexical_alpha=LEXICAL_ALPHA,
                             force_bank="working")
        p_ids = ENC.encode_ordinary(ep["prompt"])
        xp = torch.tensor([_pad(p_ids, SEQ_LEN)], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            _, retrieved = model.decode(xp, return_retrieved=True)
            logits_on = model.aux_answer_head(retrieved).float()
            logits_off = model.aux_answer_head(torch.zeros_like(retrieved)).float()
        target = torch.tensor([int(ep["answer_token_id"])], device=device)
        ce_on += F.cross_entropy(logits_on, target).item()
        ce_off += F.cross_entropy(logits_off, target).item()
        top1_on += int(logits_on[0].argmax().item() == int(ep["answer_token_id"]))
        top1_off += int(logits_off[0].argmax().item() == int(ep["answer_token_id"]))
    model.train()
    m = max(1, min(n, len(episodes)))
    return {"ce_enabled": ce_on / m, "ce_zeroed": ce_off / m,
            "top1_enabled": top1_on / m, "top1_zeroed": top1_off / m, "n": m}


def roundtrip_check(episodes: List[Dict], n: int) -> Dict[str, Any]:
    """Tokenize then decode back; confirm facts/query/answer reconstruct."""
    ok = 0
    for ep in episodes[:n]:
        good = True
        for fact in ep["facts"]:
            if ENC.decode(ENC.encode_ordinary(fact["text"])) != fact["text"]:
                good = False
                break
        if good and ENC.decode(ENC.encode_ordinary(ep["prompt"])) != ep["prompt"]:
            good = False
        # answer token must be the first token of " value" of the target fact
        tgt = ep["facts"][ep["target_fact_idx"]]
        if good and tgt.get("value"):
            exp = ENC.encode_ordinary(" " + tgt["value"])[:1]
            if not exp or exp[0] != int(ep["answer_token_id"]):
                good = False
        ok += int(good)
    m = min(n, len(episodes))
    return {"checked": m, "reconstructed": ok, "rate": ok / max(1, m)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapter validation: path + ablation")
    ap.add_argument("--adapter-dir", type=str, default=str(REPO_ROOT / "runs" / "adapter"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes-per-step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=600)
    ap.add_argument("--ablation-threshold", type=float, default=0.10)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    adapter_dir = Path(args.adapter_dir)
    train_eps = load_episodes(str(adapter_dir / "episodes_train.jsonl"))
    held_eps = load_episodes(str(adapter_dir / "episodes_heldout.jsonl"))
    stats = json.loads((adapter_dir / "adapter_stats.json").read_text(encoding="utf-8"))
    print(SEP, flush=True)
    print(f"[INFO] Loaded {len(train_eps):,} train / {len(held_eps):,} heldout episodes", flush=True)
    print(SEP, flush=True)

    StandardTransformerBlock.forward = _gc_forward
    _GC["on"] = True
    model = DCortexV2Model(DCortexConfig()).to(device)  # small config (fast validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    # ROUNDTRIP gate.
    rt = roundtrip_check(train_eps, 200)
    print(f"[INFO] ROUNDTRIP: {rt['reconstructed']}/{rt['checked']} reconstructed "
          f"({rt['rate']:.1%})", flush=True)

    # Pre-training held-out answer CE (sanity reference).
    pre = eval_answer_ce(model, held_eps, device, dtype, args.eval_episodes)
    print(f"[INFO] pre-train held-out answer CE: enabled={pre['ce_enabled']:.4f} "
          f"zeroed={pre['ce_zeroed']:.4f}", flush=True)

    print(SEP, flush=True)
    print(f"[INFO] PATH training: {args.steps} steps x {args.episodes_per_step} episodes "
          f"(small config, memory ENABLED)", flush=True)
    print(SEP, flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss_hist: List[float] = []
    eval_hist: List[Dict] = []
    order = list(range(len(train_eps)))
    random.shuffle(order)
    ptr = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        toks = 0
        for _ in range(args.episodes_per_step):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            ep = train_eps[order[ptr]]
            ptr += 1
            with torch.amp.autocast("cuda", dtype=dtype):
                total, _emit, _t1 = structural_forward(model, ep, device, zero_memory=False)
                scaled = total / args.episodes_per_step
            scaled.backward()
            model.clear_overlays()
            step_loss += float(total.item()) / args.episodes_per_step
            toks += (len(ep["facts"]) + 1) * SEQ_LEN
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_hist.append(step_loss)
        if step % 10 == 0 or step == args.steps - 1:
            dt = time.time() - t0
            eta = dt / (step + 1) * (args.steps - step - 1)
            tok_s = toks * (step + 1) / max(1e-6, dt)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Step {step:4d}/{args.steps} | train_loss {step_loss:.4f} | "
                  f"tok/s {tok_s:7.0f} | peak {peak:.2f}GB | ETA {int(eta)}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ev = eval_answer_ce(model, held_eps, device, dtype, args.eval_episodes)
            ev["step"] = step + 1
            uplift = (ev["ce_zeroed"] - ev["ce_enabled"]) / max(1e-9, ev["ce_zeroed"])
            ev["uplift"] = uplift
            eval_hist.append(ev)
            print(f"  [HELD-OUT] step={step+1} answer CE enabled={ev['ce_enabled']:.4f} "
                  f"zeroed={ev['ce_zeroed']:.4f} uplift={uplift:.1%} | "
                  f"top1 enabled={ev['top1_enabled']:.1%} zeroed={ev['top1_zeroed']:.1%}",
                  flush=True)

    final = eval_hist[-1]
    init_loss = float(np.mean(loss_hist[:10]))
    final_loss = float(np.mean(loss_hist[-10:]))

    # Gates.
    parse_rate = stats["parse_rate_of_marker_bearing"]
    g_parse = parse_rate >= 0.90
    g_round = rt["rate"] >= 0.999
    g_path = final_loss < init_loss
    uplift = final["uplift"]
    g_abl = uplift >= args.ablation_threshold and final["ce_enabled"] < final["ce_zeroed"]

    verdict = [
        {"criterion_id": "PARSE", "passed": bool(g_parse),
         "evidence": (f"{stats['counts']['parsed']:,}/{stats['counts']['marker_bearing']:,} "
                      f"marker-bearing records parsed ({parse_rate:.1%} >= 90%); "
                      f"facts/record min/median/max = {stats['facts_per_record_min']}/"
                      f"{stats['facts_per_record_median']}/{stats['facts_per_record_max']}; "
                      f"skips: zero_facts={stats['counts']['records_zero_facts']}, "
                      f"json_error={stats['counts']['skip_json_error']}.")},
        {"criterion_id": "ROUNDTRIP", "passed": bool(g_round),
         "evidence": (f"{rt['reconstructed']}/{rt['checked']} episodes reconstruct "
                      f"facts/query/answer from tokens ({rt['rate']:.1%}).")},
        {"criterion_id": "PATH_ACCEPTS", "passed": bool(g_path),
         "evidence": (f"DUAL_AGENT path ran {args.steps} steps on real episodes (no model "
                      f"change); train_loss {init_loss:.4f} -> {final_loss:.4f} (decreasing).")},
        {"criterion_id": "MEMORY_ABLATION", "passed": bool(g_abl),
         "evidence": (f"held-out answer-token CE enabled={final['ce_enabled']:.4f} vs "
                      f"zeroed={final['ce_zeroed']:.4f}; enabled is {uplift:.1%} lower "
                      f"(target >= {args.ablation_threshold:.0%}); held-out top1 enabled="
                      f"{final['top1_enabled']:.1%} vs zeroed={final['top1_zeroed']:.1%}; "
                      f"n={final['n']} unseen episodes.")},
    ]
    out = {"verdict": verdict,
           "reference": {"parse_rate": parse_rate, "roundtrip_rate": rt["rate"],
                         "train_loss_init": round(init_loss, 4),
                         "train_loss_final": round(final_loss, 4),
                         "heldout_ce_enabled": round(final["ce_enabled"], 4),
                         "heldout_ce_zeroed": round(final["ce_zeroed"], 4),
                         "heldout_ablation_uplift": round(uplift, 4),
                         "heldout_top1_enabled": round(final["top1_enabled"], 4),
                         "heldout_top1_zeroed": round(final["top1_zeroed"], 4),
                         "eval_history": eval_hist, "config": "small(175M)",
                         "validation_note": ("single-seed single-env MEASURED ablation, "
                                             "not legal-grade; >=20 seeds required to PROVE.")}}
    out_path = adapter_dir / "verdict.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(SEP, flush=True)
    for item in verdict:
        tag = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{tag}  [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(i["passed"] for i in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print("ADAPTER_VERDICT_JSON " + json.dumps(out), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
