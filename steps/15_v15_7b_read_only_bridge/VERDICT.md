# VERDICT — D_Cortex v15.7b-R Read-Only Semantic Query Bridge

**Status: BRIDGE CONTRACT VERIFIED; END-TO-END INTEGRATION BLOCKED**
**Verdict timestamp: 2026-06-15T19:12:02+03:00**
**Result: 11 PASS, 2 FAIL**

## Verified component result

The pure read-only bridge passed every contract and state-safety gate:

- accepted `QUERY_ONLY` decisions are the only routable inputs
- rejection, abstention, and mismatch preserve the original query exactly
- repeated routes are byte-identical
- the bridge exposes no mutation dependency
- bridge invocation leaves post-write memory byte-identical
- baseline and routed reads leave memory byte-identical
- all predecessor source hashes remain unchanged

## Frozen end-to-end result

| Family | Semantic route | Baseline recall | Routed recall | Uplift |
|---|---:|---:|---:|---:|
| F1 | 88.5%, wrong 0.0% | 13.0% | 13.5% | +0.5pp |
| F3 | 93.5%, wrong 0.0% | 36.0% | 36.0% | 0.0pp |
| F5 | 94.5%, wrong 0.0% | 36.5% | 36.5% | 0.0pp |

S5/S6:

- baseline honesty: `0.0% / 0.0%`
- routed honesty: `0.0% / 0.0%`
- routed overcommit: `100% / 100%`

## Gate failures

- `B7_END_TO_END_UPLIFT`: FAIL
- `B11_S5_S6_HONESTY`: FAIL

The query interpretations are accurate, but the trained neural working-memory
path does not convert them into improved answer recall and cannot abstain on
ambiguous reads.

## Diagnosis

The failure is downstream of semantic query interpretation:

1. F3/F5 use one written fact, yet recall remains only `36.0-36.5%`.
   Query routing cannot repair value emission/generalization when only one
   memory slot is present.
2. The neural answer head always emits one token, so it has no S5/S6 honesty
   mechanism.
3. A development-only `n=50` per-family cloze-address diagnostic did not
   materially change the result:
   - F1 `18% -> 24%`
   - F3 `40% -> 40%`
   - F5 `36% -> 36%`

Therefore another query prompt/router iteration is not justified.

## Evidence

- current-session artifact:
  `runs/semantic_bridge_end_to_end/results/verdict.json`
- artifact SHA-256:
  `3ca8397eacbfe47301d9a7659466403e54527271e808959667bbfb6fba0a32a2`
- frozen sample:
  `1000` trials, seed `20261315`
- sample SHA-256:
  `3667dae874cde3f9dd6c214232126430cae3e30fd5e78ec939dcc0a7bac161c4`
- historical memory checkpoint:
  `D_Cortex-main/runs/memory_campaign/results/best_model.pt`
- checkpoint timestamp:
  `2026-06-15T14:37:00+03:00`

## Claim guard

Supported:

- the read-only bridge contract is verified
- query semantics are accurate on the frozen sample
- textual query routing does not improve the measured neural-memory result
- the measured neural-memory read path lacks ambiguity honesty

Not supported:

- end-to-end semantic-memory improvement
- Pas 7a committed/provisional integration
- fact-side semantic internalization
- open-domain proof
