# VERDICT — D_Cortex v15.7b-G Explicit Referent Grounding

**Status: ABSOLUTE SAFETY TARGET MET; MILESTONE GATE FAILED**
**Verdict timestamp: 2026-06-15T19:44:57+03:00**
**Result: 8 PASS, 1 FAIL**

## Frozen result

- grounded S6 overcommit: `0/200 = 0.0%`
- grounded S6 honesty: `200/200 = 100%`
- predecessor S6 honesty on same new sample: `193/200 = 96.5%`
- honesty uplift: `+3.5pp` — FAIL frozen requirement `+5pp`
- F1/F3/F5 correct: `86.5% / 90.5% / 93.0%`, unchanged
- F1/F3/F5 wrong committed read: `0.0%`, unchanged
- S5 honesty: `100%`, unchanged

## Diagnosis

The explicit-evidence guard behaves correctly and removes all measured
pronoun-only overcommit. The failed uplift gate reflects the stronger
predecessor result on this newly frozen sample, not a regression.

The threshold is not relaxed and the seed is not rerun. Therefore the
all-gates milestone remains unsealed.

## Evidence

- current-session artifact:
  `runs/semantic_referent_grounding/results/verdict.json`
- artifact SHA-256:
  `c2284d0c348235722c8770b7cf5fd91b8c84ac63b1b23f4b843991b314adcfd7`
- sample seed: `20261500`
- sample size: `1000`

## Claim guard

Supported:

- explicit referent grounding removed all measured S6 overcommit
- it preserved known-family and S5 behavior
- it is deterministic, immutable, and has no mutation path

Not supported:

- all-gates grounding milestone
- general coreference resolution
- F1 coverage closure
- Pas 7a runtime integration
