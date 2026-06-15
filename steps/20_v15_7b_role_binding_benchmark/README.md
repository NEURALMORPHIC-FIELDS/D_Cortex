# D_Cortex v15.7b-RB0 — Non-Trivial Role-Binding Benchmark

**Status: SEALED NON-TRIVIAL BENCHMARK**
**Frozen at: 2026-06-15T19:48:00+03:00**
**Verdict at: 2026-06-15T19:49:55+03:00**

## Purpose

Construct and validate a fact-side benchmark where every known record contains
two entities and two same-attribute values, so token overlap can identify the
inventory but cannot determine the correct entity-value binding.

This directly addresses the original lexical-solvability failure before any
new fact internalizer is trained.

## Frozen sample

- `400` records per family
- seed `20261520`
- four known role-binding families:
  - RB1 direct aligned relation
  - RB2 negated crossed relation
  - RB3 rejected-then-assigned crossed relation
  - RB4 former/latter crossed relation
- one ambiguous family:
  - RB5 unresolved either/or mapping
- entities and values sourced only from sealed Pas 7a definitions
- known records require the exact set of two `(entity, attribute, value)` tuples
- ambiguous records require abstention

## Frozen lexical baselines

- ordered-first-occurrence bijection
- minimum-distance position-only bijection
- lexical Cartesian product
- safe abstain

No learned model is evaluated in this step.

## Frozen gates

| Gate | Requirement |
|---|---|
| RB0_DETERMINISTIC | repeated sample construction and hash exact |
| RB1_STRUCTURE | every known record has exactly two entities, two values, and a one-to-one truth mapping |
| RB2_AMBIGUITY | every ambiguous record has no committed truth mapping |
| RB3_NO_DUPLICATE_TEXT | all record texts unique |
| RB4_LEXICAL_NON_TRIVIAL | best non-abstaining lexical baseline exact-known rate below 50% |
| RB5_NO_TRIVIAL_SAFE_WIN | no baseline has both known exact rate at least 50% and ambiguous overcommit at most 2% |
| RB6_SAFE_ABSTAIN_COST | safe-abstain baseline has zero ambiguous overcommit and zero known coverage |
| RB7_SEALED_SOURCE | sealed Pas 7a source hash unchanged and ontology-only sourcing verified |

## Claim guard

Passing validates only that the benchmark removes the measured lexical
shortcut. It does not demonstrate a semantic fact internalizer or memory
improvement.

## Frozen verdict

All eight gates passed.

- best non-abstaining lexical/position baseline: `37.1%` exact-known
- ordered-first-occurrence: `25.0%`
- lexical Cartesian: `0.0%`
- safe abstain: `0.0%` known coverage, `0.0%` ambiguity overcommit
- sample: `2000` unique records
- sample hash:
  `7e1d681c84ceb728fa92cf04e7c463605fd1e9a2af720e01875de45d843a956a`

See [VERDICT.md](VERDICT.md).
