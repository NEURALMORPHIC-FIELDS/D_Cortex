# VERDICT — D_Cortex v15.7b-RB2 Token-Level Role-Conditioned Binder

**Status: SEALED CONTROLLED ROLE-BINDING SUBMILESTONE**
**Verdict timestamp: 2026-06-15T20:14:04+03:00**
**Result: 13 PASS, 0 FAIL**

## Frozen result

- complete role masks: `2000/2000`
- validation loss: `1.1427 -> 0.0108`, drop `99.1%`
- known test exact binding: `256/256 = 100%`
- wrong emitted mapping: `0/256 = 0%`
- ambiguous test abstention: `57/57 = 100%`
- frozen RB1 exact: `48.4%`
- RB1 uplift: `+51.6pp`
- best same-test lexical baseline: `36.7%`
- lexical uplift: `+63.3pp`
- adapter-approved provisional facts: `512/512`
- substrate byte-identical; zero trainable substrate parameters
- no relation lexicon, direct mutation path, or seal change

## Evidence

- verdict:
  `runs/semantic_role_conditioned/results/verdict.json`
- verdict SHA-256:
  `e370695d2bce6c6843e8b4514a31e3e37c3e14a78e8eabe7879d9250a1b54705`
- trained head SHA-256:
  `44d3c2b6ea4ddcba57b19905cfd031c95957a16ee18c3c6fc357bfedd7a61613`

## Interpretation

Explicit candidate-role masks plus frozen token-level contextual states remove
the measured identity-versus-swapped symmetry that defeated pooled candidate
views. The learned head can bind two entities to two values while refusing the
ambiguous family.

## Claim guard

Supported:

- controlled role binding on held-out texts and identifiers with seen syntax
  families
- complete one-to-one provisional fact emission through the adapter
- token-level role conditioning is materially better than pooled candidate text

Not supported:

- unseen-syntax or open-domain role binding
- Pas 7a ingestion, promotion, or committed-memory improvement
- end-to-end semantic-memory advantage
