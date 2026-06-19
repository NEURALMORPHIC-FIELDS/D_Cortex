# Stage I - MEASURED RESULT: STAGE_I_BINDING_FAIL (the extraction front-end is unsafe on multi-fact text)

Cert: scripts/certify_stage_i_extraction.py | verdict: runs/stage_i/results/verdict.json
Base: runs/stage_u/results/ckpt_multiattr.pt (FROZEN) | device: RTX 5080 | 5 seeds x 1500 steps
Regime: multi-entity same-attribute distinct-value pairs ("The bear is red. The fox is blue."),
linear probe at the entity token position, value_head -> frozen codebook. Held-out 10/30 entities.

## LEAD WITH THE NEGATIVE
The frozen model's per-token representation at the entity position does NOT linearly carry the
correct per-entity value when >= 2 same-attribute facts co-occur in one text.

| Metric (median over 5 seeds) | Held-out | In-sample (train ents) |
|------------------------------|----------|------------------------|
| wrong_binding (dangerous)    | 0.21     | 0.121                  |
| cross_binding (sibling's value bound to this object) | 0.1925 | ~0.12 |
| attribute_error              | 0.0225   | -                      |
| value_error (the floor)      | 0.48     | 0.353                  |
| wrong_commit_total           | 0.485    | -                      |

- G_BINDING bar was <= 0.02. Result 0.21 -> FAIL by ~10x.
- The dangerous direction (cross-binding: the SIBLING entity's value bound to this object) is ~19%
  held-out, ~12% in-sample - far above chance (~1/29 = 0.034). A systematic contamination signature.
- value_error ~0.48: about HALF the extracted facts get the wrong value. (Chance value_error ~ 0.97,
  so the probe DID learn a lot - it is right ~52% - but cannot cleanly separate co-occurring values.)

## DECOMPOSITION (the codebook is NOT the bottleneck)
- tokenizer-isolation floor (perfect extraction, gold value fed, only codebook can drift) = 0.0000.
- learned value_error = 0.48. extraction_added = 0.48 - 0.0 = 0.48.
- ALL the value error is extraction/binding-added; NONE is codebook drift. (So a VQ codebook would
  not help here - the bottleneck is upstream, at the binding/separation.)

## REPRESENTATIONAL LIMIT, not a head-generalization gap
In-sample (heads trained on these very entities) ALSO fails: wrong_binding ~0.12, value_error ~0.35.
So the frozen representation does not linearly separate co-occurring per-entity values even in-sample;
held-out only worsens it (0.21 / 0.48). Attribute routing is near-clean (~0.02) -> the entity position
IS a valid readout (type is locally bound); the specific VALUE binding is what is not linearly present.

## CAVEATS (do NOT oversell the negative)
1. Linear-probe FAIL = not LINEARLY extractable at the entity position. It does NOT prove the info is
   absent: it could be nonlinearly present, live at the value-token position, or be recoverable by a
   learned attention/query readout. Pre-declared follow-up (NOT in this cert): a shallow MLP / query
   readout to separate "not present" from "not linearly present here".
2. OOD: the base was trained on SINGLE-fact-per-encode, never multi-fact-in-one-text. This refutes
   "the CURRENT frozen rep linearly exposes co-occurring bindings", NOT "the architecture cannot be
   trained to bind". A base trained on multi-fact text might separate them.
3. The negative is specific to (frozen rep, entity-position pool, linear probe). A different readout
   may do better.

## WHAT THIS MEANS (for v1 and the vision)
- v1's CANONICAL store is untouched and still exact: the user gives the value, the token is assigned
  directly (wrong_commit 0/140). Stage I tested the EXTRACTION front-end that would AUTO-populate the
  store from raw text.
- That front-end, as a frozen-rep linear probe, is UNSAFE on multi-fact text: ~half the extracted
  facts are wrong, ~19% dangerously cross-bound. You cannot auto-extract multi-fact text into the
  honest store without one of: (a) training the base to separate co-occurring bindings, (b) a
  non-linear / attention query readout, or (c) restricting auto-extraction to single-fact inputs.
- Consistent with Stage C: single-object operations are solid (Stage U single-fact internalization
  wrong_commit=0; canonical exact), but MULTI-object separation/operation is the frontier (Stage C
  multi-distractor chaining at chance; Stage I multi-entity binding fails). The unifying thread:
  one fact = clean; keeping several apart and operating on them = unbuilt.

## NEXT (not started - architect's call)
The highest-value follow-ups, in order of decisiveness:
1. "Not-present vs not-linearly-present": shallow MLP / attention-query readout at the entity (does a
   non-linear readout recover the binding? if yes, the info is THERE, just not linear).
2. If still failing: train the base on multi-fact text (give it a reason to keep co-occurring bindings
   separated) and re-probe.
3. Pragmatic v1 path: restrict auto-extraction to single-fact inputs (where Stage U already gives a
   clean internalized value), keep multi-fact ingestion on the exact canonical path.
