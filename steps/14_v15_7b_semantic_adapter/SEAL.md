# SEAL — D_Cortex v15.7b-Q Contextual Semantic Query Internalizer

**Status: QUERY-SIDE SUBMILESTONE SEALED**
**Date: 2026-06-15**
**Verdict timestamp: 2026-06-15T18:57:17+03:00**
**Verdict: ALL 11 CONTEXTUAL GATES PASS**

## Statement

The conservative Pas 7b adapter and the frozen contextual semantic query
internalizer form a validated query-side submilestone.

The internalizer:

- uses frozen D_Cortex decoder-standard contextual states
- trains only separate semantic entity/attribute heads
- evaluates each reported F1/F3 form only in a fold where that exact form is
  absent from training
- emits only read-only query interpretations through
  `ConservativeSemanticAdapter`
- does not access or mutate memory
- leaves Pas 7a and the loaded D_Cortex substrate byte-identical

## Contextual gate results

| Gate | Result |
|---|---:|
| D0 form holdout | PASS |
| D1 F1 contextual out-of-fold | **451/528 = 85.4%** |
| D2 F3 contextual out-of-fold | **474/528 = 89.8%** |
| D3 F5 contextual final | **463/500 = 92.6%** |
| D4 wrong interpretation | F1 `0.0%`, F3 `0.0%`, F5 `0.2%` |
| D5 ambiguous honesty | **200/200 = 100% abstention** |
| D6 adapter required | PASS |
| D7 frozen substrate | PASS, byte-identical, zero trainable substrate parameters |
| D8 deterministic | PASS |
| D9 memory bypass | PASS |
| D10 Pas 7a untouched | PASS |

## Frozen evidence

- Verdict:
  `runs/semantic_contextual/results/verdict.json`
- Historical warm-start checkpoint used by the current-session run:
  `E:\proiecte_active\D_Cortex\D_Cortex-main\runs\warmstart\warmstarted_init.pt`
- Checkpoint timestamp:
  `2026-06-15T10:41:45+03:00`
- Pas 7a sealed source SHA-256:
  `25b4906ecc05a6b51b10902e54332a0ec2b26c4c622aa4e6ee74bd4961369aa3`
- `dcortex/semantic_adapter.py` SHA-256:
  `719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e`
- `dcortex/semantic_producer.py` SHA-256:
  `24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0`
- `scripts/semantic_contextual_curriculum.py` SHA-256:
  `bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57`

## Claim guard

Supported, measured in one local environment:

- a contextual query internalizer can produce conservative entity/attribute
  interpretations across the measured F1/F3/F5 forms
- wrong emitted interpretations remain at or below `0.2%`
- ambiguous queries are refused
- the adapter prevents direct commit
- the contextual path bypasses memory and leaves the substrate unchanged

Not supported:

- fact-side semantic internalization
- end-to-end F1/F3/F5 committed-memory improvement
- open-domain semantic generalization
- semantic-memory advantage over in-context systems
- proof beyond this single environment

The F1 contextual architecture was selected after development analysis on the
same sealed form definitions. The contextual pass is therefore an architecture
regression measurement, not an untouched independent proof.

## Post-seal restriction

Do not integrate this producer into Pas 7a committed memory directly.

The next permitted step requires a separately frozen bridge/evaluation that:

1. converts accepted query interpretations into read-only query routing,
2. leaves Pas 7a committed/provisional mutation rules unchanged,
3. runs the official end-to-end families,
4. reports wrong commit and ambiguity honesty before any integration claim.
