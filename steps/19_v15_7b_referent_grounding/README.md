# D_Cortex v15.7b-G — Explicit Referent Grounding Guard

**Status: ABSOLUTE SAFETY TARGET MET; FROZEN UPLIFT GATE FAILED**
**Frozen at: 2026-06-15T19:43:00+03:00**
**Verdict at: 2026-06-15T19:44:57+03:00**

## Purpose

Prevent an adapter-approved semantic query coordinate from reaching object
memory unless the selected entity has explicit evidence in the hypothesis
source text.

This is an epistemic safety layer at the internalization boundary. It is not a
synonym map, parser, entity resolver, or new classifier.

## Frozen policy

- new frozen sample: `200` trials each for F1/F3/F5/S5/S6
- sample seed: `20261500`
- predecessor: direct object reader from Step 18
- grounding rule: normalized selected-entity token sequence must occur
  contiguously in `SemanticHypothesis.source_text`
- no alias expansion or synonym mapping
- no raw query argument added to object-memory reader
- rejected grounding produces no memory read and no committed value
- F1 coverage remains explicitly outside this safety target

## Frozen gates

| Gate | Requirement |
|---|---|
| H0_PREDECESSOR_PRESERVED | Step 18 verdict artifact SHA-256 unchanged |
| H1_EXPLICIT_EVIDENCE_ONLY | no aliases; accepted grounding has exact entity-token evidence |
| H2_S6_OVERCOMMIT | grounded successor S6 overcommit at most 1% |
| H3_S6_UPLIFT | S6 honesty improves by at least 5pp over predecessor on the same new sample |
| H4_KNOWN_NO_REGRESSION | F1/F3/F5 correct rate no worse than predecessor by more than 0.5pp; wrong read at most 1% |
| H5_S5_PRESERVED | S5 honesty at least 95% and no regression beyond 0.5pp |
| H6_IMMUTABLE_DETERMINISTIC | snapshots unchanged and repeated grounded reads exact |
| H7_NO_MUTATION_PATH | grounding and reader expose no write/commit/consolidation capability |
| H8_SEALS_UNTOUCHED | Pas 7a and query-side seals unchanged |

## Claim guard

Passing would establish referent-grounding safety only. It would not close the
F1 coverage failure, prove general coreference resolution, or establish Pas 7a
runtime integration.

## Frozen verdict

- grounded S6 overcommit: `0/200 = 0.0%` — PASS
- grounded S6 honesty: `200/200 = 100%`
- frozen S6 uplift: `+3.5pp` (`96.5% -> 100%`) — FAIL required `+5pp`
- F1/F3/F5 correct and wrong-read metrics: byte-for-byte rate preservation
- S5 honesty: `100% -> 100%`
- evidence, immutability, determinism, no-mutation, and seal gates: PASS

The component remains useful as a verified absolute safety guard, but the
frozen all-gates milestone is not sealed. See [VERDICT.md](VERDICT.md).
