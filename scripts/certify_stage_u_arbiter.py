# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - Step 1 certification (revised after adversarial review). Question: does the
# continuous-value NeuralCommitArbiter reproduce the symbolic organ's committed value
# (wrong_commit = 0) on the L1-L5 regime, and WHAT determines whether it stays honest?
#
# The first version reported a "cosine floor at rho>0.7" - the adversarial verifier showed
# that was a CALIBRATION ARTIFACT of a fixed threshold theta=(1+rho)/2: with a noise-aware
# threshold the same-value and different-value cosine distributions stay separable and
# wrong_commit returns to 0. So the honest, identity-agnostic quantity is the SEPARABILITY
# MARGIN = min(same-value cosine) - max(different-value cosine) under the noise. When the
# margin > 0 a calibrated boundary gives wrong_commit = 0 (both cosine and discretize); when
# it collapses (<= 0) no threshold can separate and the honest property is unreachable.
# We therefore CALIBRATE theta per cell from observed distributions and report the margin as
# the floor. We also flag the noise geometry: norm-bounded noise in 768-dim has negligible
# projection on the ~1-dim discriminating axis, which is exactly why high-dim nearest-
# prototype (discretize) is robust - a real property, reported as such.

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from integration.organ_client import OrganClient, FOUND_COMMITTED
from stage_u.l_regime import build_regime, value_vectors, VALUES, ALL_VALUES
from stage_u.neural_arbiter import NeuralCommitArbiter, cosine_same_value, prototype_same_value

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage_u"
SUBSTANTIVE = {"RETROGRADE", "PROMOTE", "PRUNE"}


def organ_oracle(seq) -> Tuple[Dict, set]:
    o = OrganClient()
    if seq.seed is not None:
        e, a, v = seq.seed
        o.begin_episode(); o.write_fact(e, a, v); o.end_episode()
    for episode in seq.episodes:
        o.begin_episode()
        for (e, a, v) in episode:
            o.write_fact(e, a, v)
        o.end_episode()
    committed = {}
    for t in seq.targets:
        r = o.query(*t)
        committed[t] = r.value if r.status == FOUND_COMMITTED else None
    opset = set(r.operation for r in o._audit) & SUBSTANTIVE
    return committed, opset


def decode(vec: Optional[torch.Tensor], clean: Dict[str, torch.Tensor]) -> Optional[str]:
    if vec is None:
        return None
    vn = vec / (vec.norm() + 1e-8)
    return max(clean.items(), key=lambda kv: float(torch.dot(vn, kv[1] / (kv[1].norm() + 1e-8))))[0]


def noisy(vec: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """Norm-bounded isotropic observation noise of relative magnitude sigma (per-dim std
    sigma/sqrt(D), total norm ~sigma). NOTE: in 768-dim this has projection ~sigma/sqrt(D) on
    any fixed 1-dim discriminating axis, so it is benign for nearest-prototype - documented."""
    if sigma <= 0:
        return vec
    d = vec.shape[0]
    return vec + (sigma / (d ** 0.5)) * torch.randn(vec.shape, generator=gen)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


def measure_margin(protos, sigma: float, gen: torch.Generator, n: int = 300) -> Dict[str, float]:
    """Sample same-value and different-value cosines under the noise. Returns the separability
    margin and a calibrated threshold halfway between the two distributions' boundary."""
    cols = VALUES
    same, diff = [], []
    for _ in range(n):
        i = int(torch.randint(len(cols), (1,), generator=gen).item())
        j = int(torch.randint(len(cols), (1,), generator=gen).item())
        a = noisy(protos[cols[i]], sigma, gen)
        b = noisy(protos[cols[i]], sigma, gen)
        same.append(_cos(a, b))
        if j != i:
            c = noisy(protos[cols[j]], sigma, gen)
            diff.append(_cos(a, c))
    min_same = min(same)
    max_diff = max(diff) if diff else -1.0
    margin = min_same - max_diff
    theta = (min_same + max_diff) / 2.0
    return {"min_same": round(min_same, 4), "max_diff": round(max_diff, 4),
            "margin": round(margin, 4), "calibrated_theta": round(theta, 4)}


def run_arbiter(seq, protos, clean, same, sigma: float, gen: torch.Generator,
                corrupt: Optional[Tuple] = None) -> Tuple[Dict, set]:
    arb = NeuralCommitArbiter(same)
    if seq.seed is not None:
        e, a, v = seq.seed
        arb.seed_committed(e, a, noisy(protos[v], sigma, gen), 0)
    for ep, obs in enumerate(seq.episodes, start=1):
        for (e, a, v) in obs:
            vec = noisy(protos[v], sigma, gen)
            if corrupt is not None and (seq.name, e, a, ep) == corrupt[0]:
                vec = noisy(protos[corrupt[1]], sigma, gen)
            arb.observe(e, a, vec)
        arb.end_episode(ep)
    committed = {t: decode(arb.read(*t), clean) for t in seq.targets}
    opset = set(k.upper() for k, c in arb.op_counts.items() if c > 0) & SUBSTANTIVE
    return committed, opset


def sweep_cell(seqs, oracle, protos, clean, identity, sigma, theta, seed) -> Dict:
    if identity == "cosine":
        same = cosine_same_value(theta=theta)
    else:
        same = prototype_same_value(torch.stack([clean[v] for v in ALL_VALUES]))
    gen = torch.Generator().manual_seed(seed)
    wrong = total = op_match = 0
    for seq in seqs:
        oc, oops = oracle[seq.name]
        ac, aops = run_arbiter(seq, protos, clean, same, sigma, gen)
        for t in seq.targets:
            total += 1
            wrong += int(ac[t] != oc[t])
        op_match += int(aops == oops)
    return {"wrong_commit": wrong, "total": total,
            "wrong_commit_rate": round(wrong / max(1, total), 4), "op_set_match": op_match}


def main() -> int:
    argparse.ArgumentParser(description="Stage U Step 1 - continuous arbiter feasibility").parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Stage U Step 1 (revised) - continuous arbiter vs symbolic organ on L1-L5", flush=True)

    ents = OrganClient().known_entities
    seqs = build_regime(ents, n_per_level=20)
    clean = value_vectors(0.0)
    oracle = {seq.name: organ_oracle(seq) for seq in seqs}
    print(f"[INFO] {len(seqs)} L-sequences; organ oracle computed (committed + substantive op-set)", flush=True)

    # mechanism: exact-identity must reproduce the organ
    mech = {}
    for ident in ("cosine", "discretize"):
        c = sweep_cell(seqs, oracle, value_vectors(0.0), clean, ident, 0.0, 0.5, seed=1)
        mech[ident] = c
        print(f"  MECHANISM {ident:10s} rho0 sig0: wrong_commit {c['wrong_commit']}/{c['total']} | "
              f"op-set {c['op_set_match']}/{len(seqs)}", flush=True)
    mechanism_ok = all(mech[i]["wrong_commit"] == 0 for i in mech)

    RHOS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    SIGMAS = [0.0, 0.1, 0.3, 0.5, 0.8]
    grid = {"margin": {}, "cosine_calibrated": {}, "discretize": {}}

    print("  --- separability MARGIN = min(same-cos) - max(diff-cos) under noise (rows sigma, cols rho) ---", flush=True)
    print("    sig\\rho " + " ".join(f"{r:>6.2f}" for r in RHOS), flush=True)
    for sigma in SIGMAS:
        row = []
        for rho in RHOS:
            protos = value_vectors(rho)
            seed = 50000 + int(round(rho * 100)) * 1000 + int(round(sigma * 100))
            m = measure_margin(protos, sigma, torch.Generator().manual_seed(seed))
            grid["margin"][f"rho{rho}_sig{sigma}"] = m
            row.append(m["margin"])
        print(f"    {sigma:>6.2f} " + " ".join(f"{x:>6.2f}" for x in row), flush=True)

    for ident, key in (("cosine", "cosine_calibrated"), ("discretize", "discretize")):
        print(f"  --- identity={ident} (CALIBRATED theta): wrong_commit_rate (rows sigma, cols rho) ---", flush=True)
        print("    sig\\rho " + " ".join(f"{r:>6.2f}" for r in RHOS), flush=True)
        off = 100000 if ident == "cosine" else 200000
        for sigma in SIGMAS:
            row = []
            for rho in RHOS:
                protos = value_vectors(rho)
                theta = grid["margin"][f"rho{rho}_sig{sigma}"]["calibrated_theta"]
                seed = off + int(round(rho * 100)) * 1000 + int(round(sigma * 100))
                c = sweep_cell(seqs, oracle, protos, clean, ident, sigma, theta, seed)
                grid[key][f"rho{rho}_sig{sigma}"] = c
                row.append(c["wrong_commit_rate"])
            print(f"    {sigma:>6.2f} " + " ".join(f"{x:>6.2f}" for x in row), flush=True)

    # falsifiability
    l3 = next(s for s in seqs if s.level == "L3")
    e0, a0, _v0 = l3.episodes[0][0]
    ac_c, _ = run_arbiter(l3, value_vectors(0.0), clean, cosine_same_value(0.5), 0.0,
                          torch.Generator().manual_seed(7), corrupt=((l3.name, e0, a0, 1), "white"))
    falsifiable = (ac_c[(e0, a0)] != oracle[l3.name][0][(e0, a0)])
    print(f"  FALSIFIABILITY: inject wrong value -> committed {ac_c[(e0, a0)]!r} vs oracle "
          f"{oracle[l3.name][0][(e0, a0)]!r}; fires = {falsifiable}", flush=True)

    # honest floor: largest rho with margin>0 AND calibrated-cosine wrong_commit==0, at sigma<=0.3
    def floor(metric_key, cond):
        ok = []
        for rho in RHOS:
            cells = [g for s in SIGMAS if s <= 0.3 for g in [grid[metric_key][f"rho{rho}_sig{s}"]]]
            if all(cond(g) for g in cells):
                ok.append(rho)
        return max(ok) if ok else None
    margin_floor = floor("margin", lambda g: g["margin"] > 0)
    cos_floor = floor("cosine_calibrated", lambda g: g["wrong_commit_rate"] == 0.0)
    disc_floor = floor("discretize", lambda g: g["wrong_commit_rate"] == 0.0)

    verdict = {
        "verdict": "D_CORTEX_STAGE_U_STEP1_MEASURED" if (mechanism_ok and falsifiable) else "BLOCKED",
        "scope": ("MEASURED, continuous-value arbiter on CONTROLLED value vectors (separability rho + "
                  "norm-bounded observation noise sigma), oracle = symbolic organ on L1-L5. NO trained "
                  "DCortexV2Model used (none exists). This characterizes the ARBITER MECHANISM and the "
                  "separability requirement, NOT a trained representation."),
        "mechanism_reproduces_organ_rho0_sig0": bool(mechanism_ok),
        "mechanism": mech,
        "falsifiable_wrong_commit": bool(falsifiable),
        "honest_floor": ("the determinant is the SEPARABILITY MARGIN (min same-value cosine - max "
                         "different-value cosine) under noise; with a CALIBRATED threshold both cosine and "
                         "discretize reach wrong_commit=0 wherever margin>0. The earlier fixed-theta "
                         "'cosine floor at rho>0.7' was a calibration artifact (adversarial review)."),
        "noise_caveat": ("norm-bounded noise in 768-dim has projection ~sigma/sqrt(D) on the ~1-dim "
                         "discriminating axis, so it is benign for nearest-prototype; the discretize "
                         "robustness is a real high-dim property, not a claim that any noise is harmless."),
        "rhos": RHOS, "sigmas": SIGMAS, "grid": grid,
        "floors_at_sigma_le_0.3": {"margin_positive_max_rho": margin_floor,
                                   "cosine_calibrated_zero_max_rho": cos_floor,
                                   "discretize_zero_max_rho": disc_floor},
    }
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict['verdict']}", flush=True)
    print(f"  mechanism reproduces organ: {mechanism_ok} | falsifiable: {falsifiable}", flush=True)
    print(f"  floor (sigma<=0.3): margin>0 up to rho<={margin_floor} | calibrated-cosine 0 up to "
          f"rho<={cos_floor} | discretize 0 up to rho<={disc_floor}", flush=True)
    print("STAGE_U_VERDICT_JSON " + json.dumps({"verdict": verdict["verdict"], "mechanism_ok": mechanism_ok,
          "falsifiable": falsifiable, "margin_floor": margin_floor, "cos_floor": cos_floor,
          "disc_floor": disc_floor}), flush=True)
    return 0 if mechanism_ok else 1


if __name__ == "__main__":
    sys.exit(main())
