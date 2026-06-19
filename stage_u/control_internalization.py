# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - Step B decisive CONTROL. The fresh linear probe on w_value (768-dim, ~105 train
# samples) OVERFITS and is NOT trustworthy on its own; the spec requires an adversarial check.
# This module loads a trained checkpoint and runs:
#   (1) SHUFFLED-LABEL PROBE: refit the same linear probe on w_value but with the value labels
#       randomly permuted. If shuffled-label accuracy is also high, the probe is reading a
#       high-dimensional artifact (overfitting), NOT internalized value -> the probe metric is
#       discredited and DECODE must be used instead.
#   (2) DECODE (the model's OWN aux_answer_head reading w_value) is the trustworthy internalization
#       signal: it has FIXED trained weights, so it cannot overfit per-measurement. decode@alpha0
#       high = the model genuinely encodes value from context into the value vector.
#   (3) SHUFFLED-CONTEXT decode: measure decode when the (entity, value) pairing is broken
#       (value vector of one entity read against another's gold). Genuine binding -> collapses to
#       chance; a context-independent shortcut -> stays high.
# Run AFTER training, on runs/stage_u/results/ckpt_stage_u.pt.

from typing import Dict

import torch

from stage_u.observe_geometry import _collect_with_labels, VALUES
from stage_u.margin_probe import _cos


def _probe_acc(X: torch.Tensor, y: torch.Tensor, seed: int, shuffle: bool) -> float:
    n, d = X.shape
    g = torch.Generator().manual_seed(seed)
    yy = y[torch.randperm(n, generator=torch.Generator().manual_seed(seed + 99))] if shuffle else y
    perm = torch.randperm(n, generator=g)
    Xs, ys = X[perm], yy[perm]
    cut = int(0.7 * n)
    Xtr, ytr, Xte, yte = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]
    W = torch.zeros(d, len(VALUES), requires_grad=True)
    b = torch.zeros(len(VALUES), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(Xtr @ W + b, ytr).backward()
        opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(1) == yte).float().mean())


def run_controls(model, enc, lexical_alpha: float = 0.0, device="cpu", seed: int = 1234) -> Dict:
    X, y, ans_tokens = _collect_with_labels(model, enc, lexical_alpha, device)
    chance = 1.0 / len(VALUES)
    out = {"lexical_alpha": lexical_alpha, "chance": round(chance, 4)}

    # (1) probe validity: true vs shuffled-label
    out["probe_true"] = round(_probe_acc(X, y, seed, shuffle=False), 4)
    out["probe_shuffled_label"] = round(_probe_acc(X, y, seed, shuffle=True), 4)
    out["probe_overfits"] = bool(out["probe_shuffled_label"] > chance + 0.2)  # shuffled >> chance => overfit

    # (2) DECODE (model's own fixed head) - the trustworthy internalization signal
    head = getattr(model, "aux_answer_head", None)
    if head is not None:
        with torch.no_grad():
            logits = head(X.to(device)).float().cpu()
        tgt = torch.tensor([ans_tokens[VALUES[int(i)]] for i in y])
        out["decode_true"] = round(float((logits.argmax(1) == tgt).float().mean()), 4)
        # (3) shuffled-context: read each value vector against a DIFFERENT entity's gold token.
        # Genuine value-in-vector => decode still recovers the vector's OWN value (so vs shuffled
        # gold it should DROP); a global shortcut => unaffected.
        gperm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + 7))
        tgt_shuf = tgt[gperm]
        out["decode_vs_shuffled_gold"] = round(float((logits.argmax(1) == tgt_shuf).float().mean()), 4)
        # internalization is GENUINE if decode_true is high AND it collapses vs shuffled gold
        out["decode_genuine"] = bool(out["decode_true"] >= 0.80 and
                                     out["decode_vs_shuffled_gold"] <= chance + 0.1)
    return out
