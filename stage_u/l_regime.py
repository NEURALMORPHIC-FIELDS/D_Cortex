# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - L1-L5 longitudinal regime. Builds the five cross-episode consolidation
# families (promote_cycle, retrograde_only, completion, no_inflation, stale_prune), 20
# sequences each (100 total), as episode/observation event streams. Each symbolic value is
# mapped to a 768-dim vector at a CONTROLLED pairwise cosine rho (the separability knob):
#   v_i = sqrt(1-rho) * e_i + sqrt(rho) * u   (e_i orthonormal, u orthonormal to all e_i)
# so distinct values have cosine exactly rho (rho=0 orthogonal/separable, rho->1 collapse).
# The symbolic organ runs the same event stream as the ORACLE for the committed value.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

VALUES = ["red", "blue", "green", "yellow", "black", "white"]  # all in V15_COLORS (conflict set)
# extra non-color values used by L3 completion (different attributes; never conflict with colors)
ALL_VALUES = VALUES + ["big", "forest"]
DIM = 768


@dataclass
class Sequence:
    name: str
    level: str
    episodes: List[List[Tuple[str, str, str]]]          # per episode: list of (entity, attr, value)
    seed: Optional[Tuple[str, str, str]]                 # a pre-committed (entity, attr, value) or None
    targets: List[Tuple[str, str]]                       # (entity, attr) to read
    expected_committed: Dict[Tuple[str, str], Optional[str]]   # ILLUSTRATIVE; the cert scores against the
    expected_ops: Dict[str, int]                               # live organ oracle, NOT these fields. Op
    #                                                            COUNTS here are sketch values (the organ's
    #                                                            RECONCILE/PRUNE bookkeeping differs); only
    #                                                            the substantive op SET is certified.


def value_vectors(rho: float, values: List[str] = ALL_VALUES) -> Dict[str, torch.Tensor]:
    """Map each value to a 768-dim unit vector with exact pairwise cosine = rho."""
    k = len(values)
    a = (1.0 - rho) ** 0.5
    b = rho ** 0.5
    out = {}
    u = torch.zeros(DIM)
    u[k] = 1.0                                            # shared direction, orthogonal to e_i
    for i, v in enumerate(values):
        e = torch.zeros(DIM)
        e[i] = 1.0
        out[v] = a * e + b * u                            # |.|=1, <vi,vj>=rho for i!=j
    return out


def _pick(ents: List[str], idx: int) -> Tuple[str, str]:
    """Pick a target and a distinct distractor entity (both KNOWN single-token entities so the
    symbolic parser recognizes them; the neural arbiter is entity-agnostic)."""
    n = len(ents)
    e = ents[idx % n]
    d = ents[(idx + 7) % n]
    if d == e:
        d = ents[(idx + 8) % n]
    return e, d


def _seq_promote_cycle(idx: int, ents: List[str]) -> Sequence:
    e, d = _pick(ents, idx)
    v0, v1 = VALUES[idx % 6], VALUES[(idx + 1) % 6]
    return Sequence(
        name=f"L1_promote_{idx}", level="L1",
        seed=(e, "color", v0),                            # committed v0 at ep0
        episodes=[
            [(e, "color", v1), (e, "color", v1)],         # ep1: v1 twice -> RECONCILE, v1 {ep1}
            [(e, "color", v1)],                           # ep2: v1 -> v1 {ep1,ep2} -> RETROGRADE v0
            [(d, "color", VALUES[(idx + 2) % 6])],        # ep3: distractor -> ages v1 -> PROMOTE v1
        ],
        targets=[(e, "color")],
        expected_committed={(e, "color"): v1},
        expected_ops={"reconcile": 1, "retrograde": 1, "promote": 1, "prune": 0},
    )


def _seq_retrograde_only(idx: int, ents: List[str]) -> Sequence:
    e, _d = _pick(ents, idx)
    v0, v1 = VALUES[idx % 6], VALUES[(idx + 1) % 6]
    return Sequence(
        name=f"L2_retro_{idx}", level="L2",
        seed=(e, "color", v0),
        episodes=[
            [(e, "color", v1), (e, "color", v1)],         # ep1: RECONCILE, v1 {ep1}
            [(e, "color", v1)],                           # ep2: v1 {ep1,ep2} -> RETROGRADE v0, no age->no promote
        ],
        targets=[(e, "color")],
        expected_committed={(e, "color"): None},          # v0 demoted, v1 not promoted
        expected_ops={"reconcile": 1, "retrograde": 1, "promote": 0, "prune": 0},
    )


def _seq_completion(idx: int, ents: List[str]) -> Sequence:
    e, _d = _pick(ents, idx)
    return Sequence(
        name=f"L3_compl_{idx}", level="L3",
        seed=None,
        episodes=[
            [(e, "color", VALUES[idx % 6])],
            [(e, "size", "big")],
            [(e, "location", "forest")],
        ],
        targets=[(e, "color"), (e, "size"), (e, "location")],
        expected_committed={(e, "color"): VALUES[idx % 6], (e, "size"): "big", (e, "location"): "forest"},
        expected_ops={"reconcile": 0, "retrograde": 0, "promote": 0, "prune": 0},
    )


def _seq_no_inflation(idx: int, ents: List[str]) -> Sequence:
    e, d = _pick(ents, idx)
    v0, v1 = VALUES[idx % 6], VALUES[(idx + 1) % 6]
    return Sequence(
        name=f"L4_noinfl_{idx}", level="L4",
        seed=(e, "color", v0),
        episodes=[
            [(e, "color", v0), (e, "color", v0), (e, "color", v1)],   # ep1: v0,v0,v1 -> RECONCILE (v0 dedup)
            [(d, "color", VALUES[(idx + 2) % 6])],
            [(d, "color", VALUES[(idx + 3) % 6])],
            [(d, "color", VALUES[(idx + 4) % 6])],
        ],
        targets=[(e, "color")],
        expected_committed={(e, "color"): v0},            # single-episode v1 cannot overtake committed v0
        expected_ops={"reconcile": 1, "retrograde": 0, "promote": 0, "prune": 0},
    )


def _seq_stale_prune(idx: int, ents: List[str]) -> Sequence:
    e, d = _pick(ents, idx)
    v0, v1 = VALUES[idx % 6], VALUES[(idx + 1) % 6]
    return Sequence(
        name=f"L5_stale_{idx}", level="L5",
        seed=(e, "color", v0),
        episodes=[
            [(e, "color", v1)],                           # ep1: v1 challenger {ep1}
            [(d, "color", VALUES[(idx + 2) % 6])],        # ep2 distractor
            [(d, "color", VALUES[(idx + 3) % 6])],        # ep3 distractor
            [(d, "color", VALUES[(idx + 4) % 6])],        # ep4 distractor -> v1 stale (K_stale=3) -> PRUNE
        ],
        targets=[(e, "color")],
        expected_committed={(e, "color"): v0},            # v0 stays; v1 pruned
        expected_ops={"reconcile": 0, "retrograde": 0, "promote": 0, "prune": 1},
    )


_BUILDERS = {"L1": _seq_promote_cycle, "L2": _seq_retrograde_only, "L3": _seq_completion,
             "L4": _seq_no_inflation, "L5": _seq_stale_prune}


def build_regime(entities: List[str], n_per_level: int = 20) -> List[Sequence]:
    seqs: List[Sequence] = []
    for level, builder in _BUILDERS.items():
        for i in range(n_per_level):
            seqs.append(builder(i, entities))
    return seqs


# ---- symbolic organ ORACLE: run the same event stream, return committed value per target ----
def run_symbolic_oracle(seq: Sequence, organ) -> Dict[Tuple[str, str], Optional[str]]:
    """organ is an integration.organ_client.OrganClient. Returns {(entity,attr): committed_value}."""
    from integration.organ_client import FOUND_COMMITTED
    if seq.seed is not None:
        e, a, v = seq.seed
        organ.begin_episode()
        organ.write_fact(e, a, v)
        organ.end_episode()
    for episode in seq.episodes:
        organ.begin_episode()
        for (e, a, v) in episode:
            organ.write_fact(e, a, v)
        organ.end_episode()
    out = {}
    for (e, a) in seq.targets:
        reply = organ.query(e, a)
        out[(e, a)] = reply.value if reply.status == FOUND_COMMITTED else None
    return out
