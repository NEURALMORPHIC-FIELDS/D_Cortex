# D_Cortex v15.7b-RB1 — Conservative Learned Role Binder

**Status: NEGATIVE VERDICT; POOLED CANDIDATE-VIEW BRANCH STOPPED**
**Frozen at: 2026-06-15T19:53:00+03:00**

## Purpose

Train one conservative assignment scorer over frozen D_Cortex contextual
features to choose between:

1. lexical identity assignment
2. lexical swapped assignment
3. unresolved / abstain

Any selected facts must pass through the semantic adapter as
`PROVISIONAL_ONLY`.

## Frozen policy

- predecessor sample: sealed RB0 sample, unchanged
- deterministic split by text hash: 70% train / 15% validation / 15% test
- split is identifier/text holdout, **not syntax-family holdout**
- each record has three fixed candidate views: identity, swapped, unresolved
- frozen D_Cortex contextual substrate
- scalar assignment-scoring MLP, hidden width `256`
- training seed `20261530`
- margin selected on validation only from
  `{0.00, 0.05, ..., 0.50}` as the smallest value with ambiguous overcommit
  at most `2%`
- selected test mappings emit exactly two adapter-approved provisional facts
- unresolved or below-margin predictions emit nothing

## Frozen gates

| Gate | Requirement |
|---|---|
| J0_RB0_PRESERVED | RB0 verdict and sample SHA-256 unchanged |
| J1_SPLIT_SEPARATION | deterministic, disjoint train/validation/test texts |
| J2_REAL_OPTIMIZATION | validation loss drops at least 20% |
| J3_TEST_EXACT_BINDING | known test exact mapping at least 70% |
| J4_TEST_WRONG_MAPPING | wrong emitted mapping at most 5% of known test |
| J5_AMBIGUOUS_HONESTY | ambiguous test abstention at least 95% |
| J6_LEXICAL_UPLIFT | known test exact mapping exceeds same-test best lexical baseline by at least 25pp |
| J7_ADAPTER_PROVISIONAL_ONLY | every emitted fact adapter-accepted provisional-only |
| J8_FROZEN_SUBSTRATE | substrate byte-identical and zero trainable substrate parameters |
| J9_NO_DIRECT_COMMIT | binder exposes no write/commit/consolidation path |
| J10_SEALS_UNTOUCHED | Pas 7a and query-side seals unchanged |

## Claim guard

Passing would establish a controlled, development-exposed role-binding
submilestone on held-out identifiers/texts with seen syntax families. It would
not prove unseen-syntax, open-domain, Pas 7a ingestion, or committed-memory
improvement.

## Frozen verdict

- verdict timestamp: `2026-06-15T20:03:22+03:00`
- corrected artifact timestamp: `2026-06-15T20:04:29+03:00`
- result after harness-only J10 correction: `8 PASS, 3 FAIL`
- known test exact binding: `124/256 = 48.4%`
- wrong emitted mapping: `132/256 = 51.6%`
- ambiguous abstention: `57/57 = 100%`
- best same-test lexical baseline: `36.7%`
- learned uplift: `+11.7pp`
- validation loss drop: `46.2%`

The model learned the unresolved class but did not learn identity-versus-swapped
binding. No threshold was changed and the test split was not rerun.

See [`VERDICT.md`](VERDICT.md).
