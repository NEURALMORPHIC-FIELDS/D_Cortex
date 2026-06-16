# D_Cortex spine hardening: constrained closed-set attribute extraction + threshold rigor

Engineering register. MEASURED, deterministic (Qwen greedy). Distributions across 12 dev/test splits.

## Honest headline (median clears the bar; worst splits do not)
Replacing free-generation attribute parsing with CONSTRAINED closed-set classification (the model can
emit ONLY one of color/size/location/state/none, scored by option log-probability) lifted the median
F1/F3 over the 80% bar, but the worst-split floors remain below it:
- F1: median 86% (bar 80%, from 70%) | min 71% max 90% std 0.065 -> worst split 71% is BELOW 80%.
- F3: median 81% (bar 80%, from 75%) | min 67% max 90% std 0.070 -> worst split 67% is BELOW 80%.
- F0 control median 100%, F5 median 100% (no regression).
The gates are defined on the MEDIAN test score (which passes), but the hardest splits still leave F1/F3
below 80%. This is a measured improvement, not a solved problem.

## Gates
VERDICT: D_CORTEX_SPINE_HARDENING_PASS.
- G_DETERMINISM: True (greedy extraction byte-identical across 2 runs).
- G_NO_LEAK: every prompt word disjoint from the 97 F-generator aliases
  (asserted in code; the prompt uses only generic property descriptions: hue/large/small/where/condition/mood).
- G_THRESHOLD_HONEST: 12 splits, MiniLM entity-resolution threshold selected on DEV (maximize dev macro-F,
  ties->higher), reported on disjoint TEST. Selected-threshold distribution: {'min': 0.45, 'median': 0.45, 'max': 0.8}.
- G_F1_IMPROVE: median test F1 86% >= 80% (margin +6%): PASS.
- G_F3_IMPROVE: median test F3 81% >= 80% (margin +1%): PASS.
- G_NO_REGRESS: organ wrong_commit on gold 0.000, hallucinations 0, bypass leaks 0, F0 100%, F5 100%: PASS.

## Threshold sweep (resolved-known vs false-abstain-on-known)
- thr 0.3: resolved 98.8% | false-abstain 1.2%
- thr 0.35: resolved 98.3% | false-abstain 1.7%
- thr 0.4: resolved 97.8% | false-abstain 2.2%
- thr 0.45: resolved 97.0% | false-abstain 3.0%
- thr 0.5: resolved 96.8% | false-abstain 3.2%
- thr 0.55: resolved 96.8% | false-abstain 3.2%
- thr 0.6: resolved 96.8% | false-abstain 3.2%
- thr 0.65: resolved 96.8% | false-abstain 3.2%
- thr 0.7: resolved 96.8% | false-abstain 3.2%
- thr 0.75: resolved 96.8% | false-abstain 3.2%
- thr 0.8: resolved 96.8% | false-abstain 3.2%
The curve is nearly flat: most known entities extract cleanly, so the threshold trades off only a few
non-exact entities (resolved 98.8% at 0.30 -> 96.8% at 0.50+, false-abstain 1.2% -> 3.2%). The dev-
selection settled around 0.45 (median), a mild recall-favoring point.

## Error breakdown of the remaining F1/F3 failures (at median threshold)
- entity_resolution_miss: 12 (now the DOMINANT remaining cause)
- out_of_vocab_attribute (classified 'none' wrongly): 5
- wrong_attribute_choice: 2
- wrong_value: 0 (constrained value classification made ZERO value errors)
The constrained-attribute fix shifted the bottleneck OFF the attribute and ONTO entity resolution.
Value extraction is now error-free; the next lever is entity extraction, not attribute or value.

## Scope
MEASURED, symbolic organ + Qwen2.5-7B-4bit greedy (deterministic) + MiniLM, single machine. dcortex/ and steps/13 byte-identical (loaded read-only).
The improvement is genuine (F1 70->86%, F3 75->81% medians) from Qwen's semantic mapping under a generic
leak-free prompt (NOT hardcoded test aliases), but the hardest splits and entity resolution remain the
honest limitations.
