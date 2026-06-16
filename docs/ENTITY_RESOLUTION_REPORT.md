# D_Cortex spine: entity-resolution accuracy (5-way head-to-head)

Engineering register. MEASURED, deterministic. 12 dev/test splits (disjoint by fact). The honest result
is NOT a new resolver winning: the simple baseline wins, and the real lever was the INPUT.

## Honest headline (the hypothesis was refuted)
The campaign expected a better resolver to fix the entity misses. It did not. On the SOURCE TEXT, every
resolver reaches ~99-100% entity accuracy. They are STATISTICALLY TIED, not separated: R0/R1/R4 sit at
100% median and R2/R3 at 99.4% (std 0.003-0.005), a ~1-item difference across the test sets, well within
noise; R3 and R4 each reach 100% on individual splits. So there is no meaningful "winner" - the heavy
Qwen constrained classifier (R3) is NOT measurably worse than MiniLM, it is simply NOT BETTER, and it
costs much more. The fix was never a better resolver - it was the INPUT: resolving from the source text
(which contains the entity) instead of the Qwen-extracted PROPERTY word. Per the no-swap rule, the
production resolver stays MiniLM on parsimony (the heavy resolver gives no measurable gain), and only its
input changed (word -> text).

## Part 0 diagnostic (what the misses actually were)
On the current WORD-based resolver (F1/F3 query side): {'wrong_match': 0, 'no_match': 2} -> the misses are no_match
(abstains), because Qwen's free-generation entity extractor names the property word (scale, pigmentation,
dwelling, proportion) as the entity; MiniLM then correctly abstains on that word (low cosine). So no
embedder swap on that word can recover the entity; the text must be the input. (The earlier hardening
breakdown counted 12 entity misses across both fact and query sides on the word-based resolver; on the
text the count collapses to ~0.)

## 5-way head-to-head (entity accuracy, median [min/max] across 12 splits, on the text)
- R0_minilm: 100.0% [98%/100%] std 0.005
- R1_bge: 100.0% [99%/100%] std 0.003
- R2_mpnet: 99.4% [99%/100%] std 0.004
- R3: 99.4% [99%/100%] std 0.003
- R4: 100.0% [99%/100%] std 0.003
Argmax-by-median is R0_minilm (100%), but R1 and R4 also hit 100% median, so this is a TIE, not a clear
win. The 0.6pp entity gap (R3 99.4% vs R0 100%) and the 5pp F1 gap on ~20 test facts per split are both
~1-item differences within run-to-run noise (std 0.003-0.005), NOT statistically significant. The honest
conclusion is therefore: no resolver meaningfully beats another; the heavy Qwen classifier (R3) and
retrieve-rerank (R4) give NO measurable gain over the simple MiniLM baseline, so they are NOT adopted
(parsimony - same accuracy, much higher cost). Do not read "R3 is worse" into the numbers; read "R3 is no
better".

## What this fixes, and what it does NOT
- end-to-end F1 (with text-based MiniLM entity resolution): R0 median 100% (up from 86% with word-based).
- end-to-end F3: 86% for EVERY resolver (R0..R4). F3's ceiling is the ATTRIBUTE classification, NOT
  entity resolution. Entity was never the F3 bottleneck; the next lever is the F3 attribute mapping.
- F0 100%, F5 100% (no regression).

## Production change applied (validated winner)
integration/constrained_extractor.py now resolves the entity from the SOURCE TEXT via MiniLM (threshold
0.55), the validated winner R0. Verified end-to-end: F1 facts (incl. alias-confusing entities dancer/
warrior) extract correctly, and the adversarial set still abstains with ZERO grounded leaks (unknown
entities -> NONE_OBJECT; out-of-domain attributes -> PARSE_UNCERTAIN). The heavier resolvers were NOT
adopted (they did not win).

## Gates
VERDICT: D_CORTEX_ENTITY_RESOLUTION_PASS.
- G_ENTITY_DIAG: {'wrong_match': 0, 'no_match': 2} (reported).
- G_ENTITY_HEADTOHEAD: 5 resolvers, identical 12 splits, winner by median test entity accuracy = R0_minilm.
- G_ENTITY_IMPROVE: winner reduces the word-based query-side miss 2 -> 0 and median entity accuracy 100% >= 90%: True. HONEST: the diagnostic baseline here is the query-side count (2), lower
  than the both-sides hardening figure (12); either way the text-input fix drives it to ~0.
- G_NO_REGRESS: True (F0/F5/F1/F3 medians not below current; entity accuracy 100%).
- G_DETERMINISM: True. G_NO_LEAK: True (entity prompt does not list entities;
  they are scored as continuations, so prompt tokens stay disjoint from aliases and entity surfaces).

## Scope
MEASURED, symbolic organ + Qwen-4bit greedy + chosen resolver, single machine. dcortex/ and steps/13 byte-identical (loaded read-only).
The honest takeaway: entity resolution is not the limiting factor when done on the text; the simple
MiniLM baseline suffices and wins; the heavy resolvers were correctly NOT adopted; and F3's remaining
ceiling is attribute classification, not entity.
