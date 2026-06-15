# D_Cortex v15.7b-F — Conservative Semantic Fact Provisional Producer

**Status: HONEST PARTIAL; COVERAGE GATE FAILED**
**Frozen at: 2026-06-15T19:13:26+03:00**

## Purpose

Test the remaining semantic-abstraction boundary: whether the frozen
contextual D_Cortex substrate can interpret unusual fact syntax into
conservative `(entity, attribute, value)` hypotheses.

Every accepted fact hypothesis must pass through
`ConservativeSemanticAdapter` and may target only `PROVISIONAL_ONLY`. This step
does not ingest candidates into Pas 7a and cannot write committed memory.

## Frozen experiment

- source: sealed Pas 7a F1 fact constructions and standard V15 facts
- architecture: frozen contextual decoder-standard feature backend
- trainable component: separate entity, attribute, and attribute-qualified
  value heads
- four leave-one-form-out folds
- each evaluated F1 fact construction is absent from its fold training data
- fact-side margin thresholds:
  - entity `0.00`
  - attribute `0.40`
  - value `0.40`
- unknown classes on all three axes
- conflicting, multi-entity, and non-fact inputs must abstain
- no direct commit API
- Pas 7a and query-side sealed sources remain untouched

## Frozen gates

| Gate | Requirement |
|---|---|
| F0_FORM_HOLDOUT | each evaluated F1 fact form is absent from fold training and exact overlap is zero |
| F1_REAL_OPTIMIZATION | every fold lowers validation loss by at least 20% and reaches at least 95% joint validation accuracy |
| F2_FACT_OUT_OF_FOLD | aggregate held-out F1 `(entity, attribute, value)` accuracy is at least 85% |
| F3_WRONG_PROVISIONAL | wrong emitted provisional hypotheses are at most 2% |
| F4_AMBIGUOUS_HONESTY | at least 95% of frozen conflict, multi-entity, and non-fact inputs abstain |
| F5_ADAPTER_PROVISIONAL_ONLY | every emission is accepted by the adapter as `ACCEPT_PROVISIONAL` |
| F6_NO_DIRECT_COMMIT | producer and decisions expose no direct committed-memory route |
| F7_FROZEN_SUBSTRATE | contextual substrate remains byte-identical and has zero trainable parameters |
| F8_DETERMINISTIC | repeated curriculum/sample construction is exact |
| F9_SEALS_UNTOUCHED | Pas 7a and sealed query-side sources remain unchanged |
| F10_ANTI_INFLATION | repeated same-episode candidates count as one confirmation; distinct episodes remain distinct |

## Claim guard

Passing would establish only a measured fact-side provisional semantic
producer on frozen F1 forms in one local environment.

It would not establish:

- Pas 7a ingestion or promotion
- committed-memory improvement
- open-domain semantic abstraction
- end-to-end memory advantage

## First frozen verdict

All safety and honesty gates passed, but `F2_FACT_OUT_OF_FOLD` failed:

- accuracy `1199/2000 = 60.0%`
- wrong provisional `1.1%`
- ambiguity abstention `100%`

See `VERDICT.md`. The next permitted iteration constrains value decoding to the
already accepted attribute, keeps `UNKNOWN_VALUE`, and keeps all margins
unchanged.
