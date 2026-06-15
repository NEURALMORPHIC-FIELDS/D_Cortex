# VERDICT — D_Cortex v15.7b-F2 Attribute-Conditioned Fact Decoder

**Status: NEGATIVE VERDICT; BRANCH STOPPED**
**Verdict timestamp: 2026-06-15T19:28:26+03:00**
**Result: 7 PASS, 3 FAIL**

## Frozen result

| Gate | Result |
|---|---:|
| K2 conditioned accuracy | FAIL, `1211/2000 = 60.6%` |
| K3 wrong provisional | FAIL, `76/2000 = 3.8%` |
| K4 ambiguity abstention | PASS, `2400/2400 = 100%` |
| K5 accuracy uplift | FAIL, `+0.2pp` (`60.4% -> 60.6%`) |

All safety gates passed: predecessor artifact preserved, new sample held out
and deterministic, every emission adapter-accepted provisional-only, substrate
byte-identical with zero trainable parameters, sealed sources unchanged, and
no direct commit path.

## Diagnosis

Attribute conditioning is not the missing semantic mechanism. Post-hoc
renormalization can turn weak matching-attribute probabilities into confident
wrong emissions. Fold 0 wrong rate rose to `6.4%`; fold 3 rose to `8.8%`.

The measured limitation is upstream role binding, not merely the global
`attribute:value` candidate space. This branch is stopped. No thresholds were
changed.

## Evidence

- current-session artifact:
  `runs/semantic_fact_conditioned/results/verdict.json`
- artifact SHA-256:
  `d90802aaed7e77627bbba7ccb9e19ec170b32bb148131ab8f60f06fab7fa49bb`
- historical frozen substrate checkpoint timestamp:
  `2026-06-15T10:41:45+03:00`

## Claim guard

Supported:

- post-hoc attribute conditioning does not solve the measured fact-side gap
- it increases unsafe provisional emissions on two held-out syntax folds
- all memory and seal invariants remained intact

Not supported:

- fact-side semantic submilestone
- Pas 7a ingestion or promotion
- committed-memory improvement
- open-domain semantics
