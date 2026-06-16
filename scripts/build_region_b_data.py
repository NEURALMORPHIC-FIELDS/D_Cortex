# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Reproduces and PERSISTS the Region B data from commit d4e0b80 (country -> capital,
# probe-filtered to model-known facts, with model-generated clues that name neither
# the city nor the country). Qwen greedy generation is deterministic, so this
# regenerates the exact same probe-kept set and clue phrasing; persisting it makes the
# binder-validation reproducible without re-running the model. No alteration of the
# construction.

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import random
from dcortex_professional.qwen_runtime import QwenBaseModel
from dcortex_professional.region_b import load_capital_facts, probe_keep, make_clue

SEP = "=" * 70
OUT = REPO_ROOT / "data" / "professional_capable" / "region_b_data.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist Region B probe-filtered facts + clues")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe-limit", type=int, default=120)
    args = ap.parse_args()
    print(SEP, flush=True)
    print("[INFO] Reproducing Region B data (deterministic greedy, same as d4e0b80)", flush=True)
    lm = QwenBaseModel()
    if not lm.available:
        print(f"[BLOCKED] {lm.reason}", flush=True)
        return 2
    print(f"[INFO] model {lm.model_name} ({lm.precision})", flush=True)

    rng = random.Random(args.seed)
    facts = load_capital_facts()
    rng.shuffle(facts)
    kept = probe_keep(lm, facts, args.probe_limit)
    print(f"[INFO] probe-kept (model-known) facts: {len(kept)}", flush=True)
    rows = []
    for country, capital in kept:
        clue = make_clue(lm, country, capital)
        if clue:
            rows.append({"country": country, "capital": capital, "clue": clue})
    print(f"[INFO] usable facts with clues: {len(rows)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"seed": args.seed, "probe_limit": args.probe_limit,
                       "probe_kept": len(kept), "usable": len(rows), "facts": rows},
                      ensure_ascii=False, indent=2)
    OUT.write_text(body, encoding="utf-8")
    print(f"✓ persisted {len(rows)} facts+clues -> {OUT}", flush=True)
    for r in rows[:4]:
        print(f"   {r['country']} / {r['capital']} -> clue: {r['clue'][:60]}", flush=True)
    print(SEP, flush=True)
    print("REGION_B_DATA_PERSISTED " + str(len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
