# VERDICT — D_Cortex v15.7b-F Fact-Side Provisional Producer

**Status: HONEST PARTIAL; COVERAGE GATE FAILED**
**Verdict timestamp: 2026-06-15T19:22:25+03:00**
**Result: 10 PASS, 1 FAIL**

## Frozen result

- F1 fact out-of-fold accuracy: `1199/2000 = 60.0%` — FAIL
- emitted provisional hypotheses: `1221/2000`
- wrong provisional hypotheses: `22/2000 = 1.1%` — PASS
- ambiguous/non-fact abstention: `2400/2400 = 100%` — PASS
- adapter accepted provisional-only: `1221/1221` — PASS
- substrate byte-identical, zero trainable substrate parameters — PASS
- anti-inflation and predecessor seals — PASS

## Fold result

| Fold | Accuracy | Wrong provisional |
|---|---:|---:|
| 0 | 48.2% | 2.4% |
| 1 | 92.0% | 0.0% |
| 2 | 42.8% | 0.0% |
| 3 | 56.8% | 2.0% |

## Diagnosis

The producer is conservative and honest, but incomplete. Dominant abstention
reasons are:

- `VALUE_ATTRIBUTE_MISMATCH`
- `UNKNOWN_SELECTED:value`
- `MARGIN_TOO_SMALL:value`
- attribute abstention on the hardest held-out syntax

The global attribute-qualified value head redundantly predicts the attribute
inside the value ID after the separate attribute head has already selected it.
The next justified experiment constrains value decoding to the accepted
attribute while retaining `UNKNOWN_VALUE` and unchanged margins.

## Evidence

- current-session artifact:
  `runs/semantic_fact_curriculum/results/verdict.json`
- artifact SHA-256:
  `185e8f102449c69b2a1bad08afde475ac8319aed23da15dae41cd712499185d3`
- historical query substrate checkpoint timestamp:
  `2026-06-15T10:41:45+03:00`

## Claim guard

Supported:

- an honest provisional-only fact semantic producer exists
- wrong provisional rate remains below 2%
- ambiguity is refused
- coverage remains insufficient

Not supported:

- fact-side submilestone seal
- Pas 7a ingestion/promotion
- committed-memory improvement
- open-domain semantics
