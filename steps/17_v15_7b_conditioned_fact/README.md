# D_Cortex v15.7b-F2 — Attribute-Conditioned Provisional Fact Decoder

**Status: NEGATIVE VERDICT; POST-HOC CONDITIONING STOPPED**
**Frozen at: 2026-06-15T19:22:40+03:00**
**Verdict at: 2026-06-15T19:28:26+03:00**

## Purpose

Remove one measured representational contradiction from the fact-side
producer: after the attribute head passes its unchanged margin, value decoding
must rank only values belonging to that attribute plus `UNKNOWN_VALUE`.

The underlying heads, contextual substrate, adapter, and margins remain
unchanged. The successor is evaluated on a newly frozen sample.

## Frozen policy

- predecessor heads: `runs/semantic_fact_curriculum/results/fold_*_best_head.pt`
- new held-out sample: `500` F1 facts per fold, seeds `20261450..20261453`
- new ambiguity sample: `600` records, seed `20261460`
- entity margin `0.00`
- attribute margin `0.40`
- conditioned value margin `0.40`
- `UNKNOWN_VALUE` always remains a candidate
- selected value must still agree with selected attribute
- all emissions remain adapter `PROVISIONAL_ONLY`

## Frozen gates

| Gate | Requirement |
|---|---|
| K0_PREDECESSOR_PRESERVED | predecessor verdict artifact SHA-256 unchanged |
| K1_NEW_SAMPLE_HOLDOUT | new sample deterministic and absent from fold training |
| K2_CONDITIONED_ACCURACY | conditioned out-of-fold accuracy at least 85% |
| K3_WRONG_PROVISIONAL | conditioned wrong provisional rate at most 2% |
| K4_AMBIGUOUS_HONESTY | conditioned ambiguity abstention at least 95% |
| K5_COVERAGE_UPLIFT | conditioned accuracy exceeds unconstrained producer on the same new sample by at least 15pp |
| K6_ADAPTER_PROVISIONAL_ONLY | every conditioned emission is adapter-accepted provisional-only |
| K7_FROZEN_SUBSTRATE | substrate byte-identical and zero trainable parameters |
| K8_SEALS_UNTOUCHED | Pas 7a and query-side sealed sources unchanged |
| K9_NO_DIRECT_COMMIT | conditioned producer exposes no direct commit path |

## Claim guard

This is an architecture regression on the same F1 form families with a new
sample, not an independent open-domain proof. Passing would justify a narrow
fact-side provisional submilestone only.

## Frozen verdict

- `K2_CONDITIONED_ACCURACY`: FAIL, `1211/2000 = 60.6%`
- `K3_WRONG_PROVISIONAL`: FAIL, `76/2000 = 3.8%`
- `K5_COVERAGE_UPLIFT`: FAIL, `+0.2pp` (`60.4% -> 60.6%`)
- ambiguity abstention: PASS, `2400/2400 = 100%`
- adapter provisional-only, frozen substrate, seals, and no-direct-commit: PASS

Post-hoc renormalization promoted weak within-attribute alternatives into
confident wrong emissions. This branch is stopped without threshold changes.
See [VERDICT.md](VERDICT.md).
