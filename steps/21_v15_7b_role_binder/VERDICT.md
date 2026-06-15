# VERDICT — D_Cortex v15.7b-RB1 Conservative Learned Role Binder

**Status: NEGATIVE; POOLED CANDIDATE-VIEW BRANCH STOPPED**
**Verdict timestamp: 2026-06-15T20:03:22+03:00**
**Harness-only correction: 2026-06-15T20:04:29+03:00**
**Result after correction: 8 PASS, 3 FAIL**

## Frozen result

- validation loss: `1.0985 -> 0.5909`, drop `46.2%`
- validation best accuracy: `61.2%`
- known test exact binding: `124/256 = 48.4%` — FAIL
- wrong emitted mapping: `132/256 = 51.6%` — FAIL
- ambiguous test abstention: `57/57 = 100%` — PASS
- best same-test lexical baseline: `36.7%`
- learned uplift: `+11.7pp` — FAIL
- adapter provisional-only facts: `512/512`
- substrate byte-identical; zero trainable substrate parameters

## Family diagnosis

| Family | Exact binding |
|---|---:|
| RB1 | `30/61 = 49.2%` |
| RB2 | `34/65 = 52.3%` |
| RB3 | `26/63 = 41.3%` |
| RB4 | `34/67 = 50.7%` |
| RB5 ambiguity | `57/57 = 100%` abstained |

The scalar scorer learned whether a source was unresolved, but identity and
swapped candidates remained effectively symmetric. Five-view pooled candidate
text does not expose sufficient token-level relational evidence.

## Harness correction

The first generated verdict marked J10 false because the expected SHA-256
literal for `dcortex/semantic_adapter.py` omitted one zero. The actual file hash
was unchanged. Only J10 and its metadata were corrected; training, calibration,
test predictions, thresholds, and the frozen split were not rerun.

- original artifact SHA-256:
  `e6bba804dccc437dfe3d76d9b306f626d1db51f2cb2d9dab46c4e5877c743ebf`
- corrected artifact SHA-256:
  `92035e5d8148ead5d03d3ba5fac571bc646103ee1acd8a2677216e07d30c0b6f`
- trained head SHA-256:
  `0ee30342cf9ca9b1b942e16b60b7b684489fa93f9bbd93fcc1f79346d333313d`

## Claim guard

Supported:

- pooled complete-candidate scoring is insufficient for measured role binding
- unresolved detection and provisional-only safety work
- token-level candidate-conditioned relation evidence is justified

Not supported:

- semantic role-binding capability
- unseen-syntax generalization
- Pas 7a ingestion or committed-memory improvement
