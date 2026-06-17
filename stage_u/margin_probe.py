# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - Step 2 instrument: measure the SEPARABILITY MARGIN of DCortexV2Model's stored
# VALUE representation against the Step 1 bar. For each value word, in several contexts
# (entity x template), we run encode() and take the writer's stored value vector
# (_enc_aux["w_value"]); the margin is min(same-value cosine across contexts) -
# max(different-value cosine). margin > 0 means the arbiter (Step 1) would keep wrong_commit
# = 0 on this representation. We sweep lexical_alpha: alpha=0.9 is the operative stored value
# (lexically bound to the answer token); alpha=0 is the PURELY CONTEXTUAL value (the hard
# "internalization" case - the model must separate values from context, not copy the token).
# This is a MEASUREMENT (no training, no constraint).

from typing import Dict, List

import torch

VALUES = ["red", "blue", "green", "yellow", "black", "white"]
ENTITIES = ["bear", "dog", "cat", "fox", "wolf"]
TEMPLATES = ["The {e} is {v}.", "The {e} looked {v}.", "A {v} {e} stood nearby.",
             "The {e} was painted {v}.", "Everyone saw the {v} {e}."]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


def collect_values(model, enc, lexical_alpha: float, device="cpu") -> Dict[str, List[torch.Tensor]]:
    """Return {value: [stored w_value vectors across contexts]} for one lexical_alpha."""
    out: Dict[str, List[torch.Tensor]] = {v: [] for v in VALUES}
    with torch.no_grad():
        for v in VALUES:
            ans_tok = enc.encode_ordinary(" " + v)[0]
            for e in ENTITIES:
                for tpl in TEMPLATES:
                    text = tpl.format(e=e, v=v)
                    ids = torch.tensor([enc.encode_ordinary(text)], device=device)
                    ans = torch.tensor([ans_tok], device=device)
                    if hasattr(model, "reset_memory"):
                        model.reset_memory()
                    aux = model.encode(ids, answer_token_id=ans, lexical_alpha=lexical_alpha)
                    out[v].append(aux["w_value"][0].detach().float().cpu())
    return out


def margin(values: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    """Separability margin = min(same-value cosine) - max(different-value cosine)."""
    same, diff = [], []
    vals = list(values)
    for vi in vals:
        vs = values[vi]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                same.append(_cos(vs[i], vs[j]))
    for a in range(len(vals)):
        for b in range(a + 1, len(vals)):
            for xa in values[vals[a]]:
                for xb in values[vals[b]]:
                    diff.append(_cos(xa, xb))
    min_same = min(same) if same else 1.0
    max_diff = max(diff) if diff else -1.0
    import statistics
    return {"min_same": round(min_same, 4), "mean_same": round(statistics.mean(same), 4),
            "max_diff": round(max_diff, 4), "mean_diff": round(statistics.mean(diff), 4),
            "margin": round(min_same - max_diff, 4),
            "mean_gap": round(statistics.mean(same) - statistics.mean(diff), 4)}
