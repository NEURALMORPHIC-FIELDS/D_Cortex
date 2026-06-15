# VERDICT — D_Cortex v15.7b-O Direct Semantic Object Read

**Status: HONEST PARTIAL**
**Verdict timestamp: 2026-06-15T19:39:32+03:00**
**Result: 8 PASS, 2 FAIL**

## Frozen result

| Family / gate | Result |
|---|---:|
| F1 direct correct read | `169/200 = 84.5%` — FAIL threshold 85% |
| F3 direct correct read | `182/200 = 91.0%` |
| F5 direct correct read | `187/200 = 93.5%` |
| F1/F3/F5 wrong committed read | `0/200` each |
| S5 disputed/refused honesty | `200/200 = 100%`, overcommit `0%` |
| S6 no-committed-value honesty | `187/200 = 93.5%` — FAIL |
| S6 overcommit | `13/200 = 6.5%` |

Direct-coordinate, immutable snapshot, accepted-query-only, no-mutation,
determinism, substrate, and seal gates all passed.

## Diagnosis

The direct object reader is not the source of known-family errors: every
approved correct coordinate returns the correct committed value, and no wrong
committed read occurred.

S6 exposes missing referent grounding upstream. Pronoun-only queries received
arbitrary approved entity coordinates. All thirteen overcommits selected
`crown` or `scroll` despite neither appearing in the query.

## Evidence

- current-session artifact:
  `runs/semantic_object_read/results/verdict.json`
- artifact SHA-256:
  `13d0df32d6d4de17446c7a09dddf048108866cc051a5a95fd7058ff5eb63efa2`
- sample seed: `20261480`
- sample size: `1000`

## Claim guard

Supported:

- direct semantic-coordinate object reading is deterministic and immutable
- known reads have zero wrong committed values on this sample
- S5 conflict state is read honestly
- explicit referent grounding is required before memory access

Not supported:

- complete object-read submilestone
- Pas 7a runtime integration
- fact-side internalization
- open-domain semantics
