# VERDICT — D_Cortex v15.7b-RB3 Leave-One-Syntax-Family-Out

**Status: NEGATIVE GENERALIZATION VERDICT; MEMORY INTEGRATION BLOCKED**
**Verdict timestamp: 2026-06-15T20:20:25+03:00**
**Result: 9 PASS, 3 FAIL**

## Frozen result

- all four held-out families absent from train and validation: PASS
- complete role masks: `2000/2000`
- validation loss drop at least `98.9%` in every fold
- aggregate unseen-syntax exact: `911/1600 = 56.9%` — FAIL
- aggregate wrong emitted mapping: `503/1600 = 31.4%` — FAIL
- minimum family exact: `39.2%` — FAIL
- ambiguity abstention: `100%` in every fold
- best aggregate lexical baseline: `37.1%`
- lexical uplift: `+19.9pp` — FAIL
- all emitted facts adapter-approved provisional-only
- substrate and seals unchanged

## Per-family result

| Held-out family | Exact | Wrong | Abstained |
|---|---:|---:|---:|
| RB1 | `251/400 = 62.8%` | `97/400 = 24.3%` | `52/400 = 13.0%` |
| RB2 | `157/400 = 39.2%` | `243/400 = 60.8%` | `0/400 = 0.0%` |
| RB3 | `317/400 = 79.3%` | `83/400 = 20.8%` | `0/400 = 0.0%` |
| RB4 | `186/400 = 46.5%` | `80/400 = 20.0%` | `134/400 = 33.5%` |

## Evidence

- verdict:
  `runs/semantic_syntax_holdout/results/verdict.json`
- verdict SHA-256:
  `a6551156e655b714ceb7360c9792c55a69ffde4de20944c0757011dba8cd22c4`

## Interpretation

RB2 is a valid seen-syntax role binder, but it does not yet generalize safely
to unseen constructions. Perfect validation optimization and perfect ambiguity
honesty do not compensate for the measured `31.4%` wrong-mapping rate.

Memory ingestion is therefore blocked. Another architecture tweak on the same
four families would be development overfitting, not evidence.

## Claim guard

Supported:

- partial construction-family generalization above lexical baseline
- seen-syntax RB2 capability does not transfer safely enough to unseen syntax
- independent construction diversity is now the limiting prerequisite

Not supported:

- safe semantic fact ingestion
- open-domain role binding
- Pas 7a or committed-memory improvement
