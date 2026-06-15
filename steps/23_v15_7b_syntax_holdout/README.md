# D_Cortex v15.7b-RB3 — Leave-One-Syntax-Family-Out Role Binding

**Status: NEGATIVE GENERALIZATION VERDICT; MEMORY INTEGRATION BLOCKED**
**Frozen at: 2026-06-15T20:17:00+03:00**

## Purpose

Determine whether the sealed RB2 token-level role-conditioned architecture
generalizes relational binding to an unseen construction family or only learns
the four development-exposed family templates.

## Frozen evaluation

- predecessor RB0 sample unchanged
- four folds; each fold holds out exactly one of `RB1`, `RB2`, `RB3`, `RB4`
  from both training and validation
- held-out family: all 400 records used only for that fold's test
- seen-family training and validation use the frozen RB1 text-hash split
- RB5 ambiguity uses frozen RB1 train/validation/test partitions
- RB2 architecture and hyperparameters unchanged
- one fresh head per fold; same frozen contextual token states and role masks
- margin selected separately per fold on validation only from
  `{0.00, 0.05, ..., 0.50}`, smallest value with ambiguity overcommit at most
  `2%`
- no test-family threshold changes or reruns

## Frozen gates

| Gate | Requirement |
|---|---|
| M0_PREDECESSORS_PRESERVED | RB0, RB1, and RB2 verdict artifacts unchanged |
| M1_SYNTAX_FAMILY_HOLDOUT | held-out family absent from train and validation in every fold |
| M2_ROLE_MASK_INTEGRITY | exact candidate inventory only; no truth input |
| M3_REAL_OPTIMIZATION | validation loss drops at least 20% in every fold |
| M4_UNSEEN_SYNTAX_BINDING | aggregate held-out exact at least 70% and every family at least 55% |
| M5_WRONG_MAPPING | aggregate wrong emitted mapping at most 10% |
| M6_AMBIGUOUS_HONESTY | ambiguous test abstention at least 95% in every fold |
| M7_LEXICAL_UPLIFT | aggregate held-out exact exceeds aggregate lexical baseline by at least 25pp |
| M8_ADAPTER_PROVISIONAL_ONLY | every emitted fact adapter-approved provisional-only |
| M9_FROZEN_SUBSTRATE | substrate byte-identical and zero trainable substrate parameters |
| M10_NO_HANDWRITTEN_RELATION_OR_COMMIT | no relation lexicon/rules and no direct mutation path |
| M11_SEALS_UNTOUCHED | Pas 7a and query-side seals unchanged |

## Claim guard

Passing would establish leave-one-construction-family-out generalization over
four controlled synthetic families. It would still not prove open-domain
language understanding, Pas 7a ingestion, committed-memory improvement, or
end-to-end semantic-memory advantage.

## Frozen verdict

- verdict timestamp: `2026-06-15T20:20:25+03:00`
- result: `9 PASS, 3 FAIL`
- aggregate unseen-syntax exact: `911/1600 = 56.9%`
- aggregate wrong emitted mapping: `503/1600 = 31.4%`
- minimum family exact: `39.2%`
- ambiguity abstention: `100%` in every fold
- lexical uplift: `+19.9pp`

The architecture generalizes partially, but not safely enough for memory
ingestion. No fold, threshold, or test family was rerun.

See [`VERDICT.md`](VERDICT.md).
