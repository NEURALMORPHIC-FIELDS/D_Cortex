# D_Cortex v15.7b-RB2 — Token-Level Role-Conditioned Binder

**Status: SEALED CONTROLLED ROLE-BINDING SUBMILESTONE**
**Frozen at: 2026-06-15T20:09:00+03:00**

## Purpose

Test the specific RB1 failure diagnosis: complete-candidate pooled features
detect ambiguity but lose the token-level relation needed to distinguish
identity from swapped assignments.

RB2 exposes frozen D_Cortex contextual token states to a learned
candidate-conditioned sequence scorer. It adds role masks only:

- `NONE`
- `ENTITY_A`
- `VALUE_A`
- `ENTITY_B`
- `VALUE_B`

It adds no synonym map, relation lexicon, syntax-family rules, memory mutation,
or direct commit path.

## Frozen architecture

- predecessor sample and deterministic RB1 split unchanged
- exact source mentions are marked from the supplied entity/value inventory
- identity and swapped candidates differ only in value-role assignment
- unresolved candidate has no role marks
- frozen final D_Cortex decoder-standard token states
- trainable projection width `128`
- trainable role embedding width `32`
- one bidirectional GRU layer, hidden width `128` per direction
- masked attention pooling to one scalar per candidate
- training seed `20261540`
- validation-only margin selection on `{0.00, 0.05, ..., 0.50}`, smallest
  threshold with ambiguous overcommit at most `2%`
- selected mappings emit exactly two adapter-approved `PROVISIONAL_ONLY` facts

## Frozen gates

| Gate | Requirement |
|---|---|
| L0_PREDECESSORS_PRESERVED | RB0 and RB1 verdict artifacts unchanged |
| L1_SPLIT_SEPARATION | deterministic disjoint RB1 train/validation/test texts |
| L2_ROLE_MASK_INTEGRITY | every supplied mention marked; swap changes value roles only; no truth label input |
| L3_REAL_OPTIMIZATION | validation loss drops at least 20% |
| L4_TEST_EXACT_BINDING | known test exact mapping at least 75% |
| L5_TEST_WRONG_MAPPING | wrong emitted mapping at most 5% of known test |
| L6_AMBIGUOUS_HONESTY | ambiguous test abstention at least 95% |
| L7_RB1_UPLIFT | known exact mapping improves at least 20pp over frozen RB1 |
| L8_LEXICAL_UPLIFT | known exact mapping exceeds same-test best lexical baseline by at least 30pp |
| L9_ADAPTER_PROVISIONAL_ONLY | every emitted fact adapter-approved provisional-only |
| L10_FROZEN_SUBSTRATE | substrate byte-identical and zero trainable substrate parameters |
| L11_NO_HANDWRITTEN_RELATION_OR_COMMIT | no relation lexicon/rule map and no direct mutation path |
| L12_SEALS_UNTOUCHED | Pas 7a and query-side seals unchanged |

## Claim guard

Passing would establish a controlled, development-exposed token-level
role-binding submilestone on held-out texts and identifiers with seen syntax
families. It would not prove unseen syntax, open-domain grounding, Pas 7a
ingestion, committed-memory improvement, or end-to-end semantic-memory
advantage.

## Frozen verdict

- verdict timestamp: `2026-06-15T20:14:04+03:00`
- result: `13/13 PASS`
- known test exact binding: `256/256 = 100%`
- wrong emitted mapping: `0/256 = 0%`
- ambiguous abstention: `57/57 = 100%`
- RB1 uplift: `+51.6pp`
- same-test lexical uplift: `+63.3pp`
- validation loss drop: `99.1%`

See [`VERDICT.md`](VERDICT.md).
