# Stage U - Step B: the decisive internalization experiment (SPEC, ready to launch cold)

Status: SPEC ONLY. Do NOT launch as an open long run at the end of a long session. Run it
fresh, bounded, with the pre-declared target below.

## Why B (the reframing that decides it)
Step 2 measured (runs/stage_u/results/organic_geometry.json, well-trained model top1->1.0):
- alpha=0.9 (lexically-bound value): fully recoverable (linear probe 1.0, decode-head 1.0).
- alpha=0.0 (pure CONTEXTUAL value): linear probe 0.356, decode-head 0.167 (chance), AND it
  gets WORSE with training (probe 0.56 -> 0.36, decode 0.33 -> 0.17 across the run).

So the model organically COPIES the answer token (lexical binding) and does NOT internalize the
value from context - and gradient descent moves it FURTHER from internalization, not toward it.
The vision ("memory as the organ of thought") requires the value to be INTERNALIZED, not a copied
token. Therefore:
- A (close U on the operative value) gives an honest arbiter over PARROTED tokens - the mechanism
  is real, the think-in-memory premise is empty. A abandons the vision.
- B is the decisive test of the vision's central premise: can the model be FORCED to internalize
  (encode value from context) instead of copying? The current answer is discouraging; B is exactly
  the experiment that settles it.

Decision (owner): B - finish the vision, not the mechanism milestone.

## The experiment
Train DCortexV2Model in a COPY-PROOF regime - lexical_alpha LOW during TRAINING (not merely
measured at 0.0 on a lexically-trained model). The architecture makes alpha=0.0 copy-proof: the
answer is decoded ONLY from the written value vector (aux_answer_head(retrieved_value)), and at
alpha=0.0 that value carries NO lexical component, so to minimize L_emit the encoder MUST encode
value-identity into the value from context. There is no token to copy.

Regimes (run both, in order):
1. ANNEAL: lexical_alpha 0.5 -> 0.0 linearly over the first half of training (bootstrap, then
   remove the crutch). Gives internalization its best chance.
2. STRICT: lexical_alpha = 0.0 from step 0 (no crutch ever). The hard test.

## Metrics tracked vs training step (the falsifiable instrument)
At the TRAINING alpha (i.e. measure at alpha=0.0 in the copy-proof regime), every K steps:
- top1: answer accuracy (end-to-end internalize + retrieve + decode).
- linear_probe@alpha0: is value-identity linearly recoverable from the written value vector
  (the internalization signal - this is the number that must RISE).
- decode_head@alpha0: does the model's own aux_answer_head recover the value.
Baselines on the same axes: natural training at alpha=0.9 gave probe@alpha0 FALLING 0.56 -> 0.36;
chance = 1/15 = 0.067.

## Pre-declared falsifiable targets (declare BEFORE running; no moving the bar after)
- INTERNALIZATION ACHIEVED (vision realizable on this substrate):
  probe@alpha0 RISES with training AND reaches >= 0.80, AND top1@alpha0 >= 0.80, by the step
  budget. (Contrast: natural training FELL to 0.36.)
- INTERNALIZATION REFUTED (deep finding - vision needs a different architecture, not just
  environment shaping): probe@alpha0 stays < 0.50 / plateaus / falls even under the copy-proof
  regime, OR top1@alpha0 cannot exceed ~0.50 (the model cannot solve the task without the lexical
  crutch). 
- PARTIAL: probe in [0.50, 0.80) - report as such, no rounding up.
Both ACHIEVED and REFUTED are maximally valuable: they tell the owner whether the vision is
realizable on DCortexV2Model as-is, or requires an architectural change.

## Budget (bounded - this is not an open run)
- Fixed step budget per regime: 1500 steps (extend to 3000 only if probe is RISING but not yet at
  0.80 by 1500 - a single pre-authorized extension, not open-ended).
- grad_accum 16, seq_len 32, 3 facts/episode (the fast-but-well-training config from step 2;
  reaches top1 1.0 at alpha=0.9 by ~step 400, so 1500 is ample for the harder alpha=0.0 task).
- measure-every 150 (to see the trajectory, not just endpoints).
- GPU note: training is batch=1 (memory-stateful) -> ~20% GPU util regardless of device; ~50
  steps/min on the RTX 5080. 1500 steps ~ 30 min. Budget accordingly; do not relaunch on CPU.

## Harness (already enabled, ready to launch)
scripts/train_stage_u.py now accepts:
  --lexical-alpha FLOAT      (default 0.9; set 0.0 for STRICT)
  --anneal-alpha FROM TO     (e.g. --anneal-alpha 0.5 0.0 for ANNEAL over the first half)
  --measure-alpha FLOAT      (the alpha at which probe/top1 are measured; set 0.0 for B)
It trains with the (possibly annealed) training alpha and reports probe@measure-alpha + top1 each
measure-every, writing runs/stage_u/results/train_trajectory.json + organic_geometry.json + a
checkpoint. Launch commands (cold, fresh session):
  # STRICT (the hard test)
  python scripts/train_stage_u.py --steps 1500 --lexical-alpha 0.0 --measure-alpha 0.0 --measure-every 150
  # ANNEAL (best chance)
  python scripts/train_stage_u.py --steps 1500 --anneal-alpha 0.5 0.0 --measure-alpha 0.0 --measure-every 150

## Anti-confabulation discipline (mandatory)
- Deterministic (fixed seeds); report the FULL trajectory, not just the best point.
- Lead with the negative: if probe falls/plateaus, say REFUTED first.
- The probe is the gate; top1 alone is NOT internalization (the model could still find a non-value
  shortcut - the probe on the value vector is what proves value-identity lives in the value).
- Adversarially verify any "internalization achieved" before claiming it (e.g. shuffled-context
  control: does probe collapse when the fact context is shuffled? if not, the probe is reading a
  trivial artifact, not internalized value).
- Single model, small synthetic regime, single machine: NOT a generality claim.

## What each outcome means for the project
- ACHIEVED -> B is the path; wire the honest-mechanics arbiter on the internalized value; the
  vision is realizable on DCortexV2Model.
- REFUTED -> the substrate cannot internalize even when forced; the vision needs an architecture
  change (the value-binding mechanism itself), not environment shaping. This redirects the whole
  program and is the single most decision-relevant thing to know.
