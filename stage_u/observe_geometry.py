# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - observe the model's ORGANIC value-identity geometry. Methodology pivot: do NOT
# force the model's stored value to be cosine-separable (our rule); instead OBSERVE where the
# trained model naturally puts value-identity, then build the honest mechanics to live in
# THAT space. For the value set across contexts we collect the writer value (w_value) and ask,
# in several candidate spaces, how recoverable the value identity is:
#   (1) raw_cosine     - separability margin on w_value cosine (the space we were wrongly forcing)
#   (2) decode_head    - does the model's OWN aux_answer_head read the value back from w_value?
#                        (same-value -> same predicted token; different-value -> different)
#   (3) linear_probe   - is value LINEARLY decodable from w_value (info present but not in cosine)?
#   (4) head_logit_cos - separability margin in the aux_answer_head LOGIT space (post-readout)
# This is OBSERVATION (no training, no constraint).

from typing import Dict, List

import torch

from stage_u.margin_probe import VALUES, ENTITIES, TEMPLATES, _cos, collect_values, margin


def _collect_with_labels(model, enc, lexical_alpha: float, device="cpu"):
    """Return (X [N,768] w_value, y [N] value-index, ans_tokens dict)."""
    X, y = [], []
    ans_tokens = {v: enc.encode_ordinary(" " + v)[0] for v in VALUES}
    with torch.no_grad():
        for vi, v in enumerate(VALUES):
            for e in ENTITIES:
                for tpl in TEMPLATES:
                    ids = torch.tensor([enc.encode_ordinary(tpl.format(e=e, v=v))], device=device)
                    ans = torch.tensor([ans_tokens[v]], device=device)
                    if hasattr(model, "reset_memory"):
                        model.reset_memory()
                    aux = model.encode(ids, answer_token_id=ans, lexical_alpha=lexical_alpha)
                    X.append(aux["w_value"][0].detach().float().cpu())
                    y.append(vi)
    return torch.stack(X), torch.tensor(y), ans_tokens


def observe(model, enc, lexical_alpha: float = 0.9, device="cpu", seed: int = 0) -> Dict:
    X, y, ans_tokens = _collect_with_labels(model, enc, lexical_alpha, device)
    n, d = X.shape
    out: Dict[str, object] = {"lexical_alpha": lexical_alpha, "n": n}

    # (1) raw cosine margin
    vals = collect_values(model, enc, lexical_alpha, device)
    out["raw_cosine_margin"] = margin(vals)["margin"]

    # (2) decode via the model's OWN aux_answer_head: w_value -> answer-token argmax
    head = getattr(model, "aux_answer_head", None)
    if head is not None:
        with torch.no_grad():
            logits = head(X.to(device)).float().cpu()           # [N, vocab]
        pred = logits.argmax(dim=1)
        tgt = torch.tensor([ans_tokens[VALUES[int(i)]] for i in y])
        out["decode_head_value_accuracy"] = round(float((pred == tgt).float().mean()), 4)
        # head logit-space separability margin (cosine on the readout vectors)
        groups = {vi: [logits[i] for i in range(n) if int(y[i]) == vi] for vi in range(len(VALUES))}
        same, diff = [], []
        for vi, gs in groups.items():
            for a in range(len(gs)):
                for b in range(a + 1, len(gs)):
                    same.append(_cos(gs[a], gs[b]))
        ks = list(groups)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                for xa in groups[ks[a]]:
                    for xb in groups[ks[b]]:
                        diff.append(_cos(xa, xb))
        out["head_logit_cosine_margin"] = round((min(same) if same else 1) - (max(diff) if diff else -1), 4)

    # (3) linear probe: is value LINEARLY decodable from w_value? (held-out by context)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    Xs, ys = X[perm], y[perm]
    cut = int(0.7 * n)
    Xtr, ytr, Xte, yte = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]
    W = torch.zeros(d, len(VALUES), requires_grad=True)
    b = torch.zeros(len(VALUES), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = float(((Xte @ W + b).argmax(1) == yte).float().mean())
    out["linear_probe_accuracy"] = round(acc, 4)
    out["chance"] = round(1.0 / len(VALUES), 4)
    return out
