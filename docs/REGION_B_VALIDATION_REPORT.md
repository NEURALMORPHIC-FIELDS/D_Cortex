# Region B binder validation: held-out + ablation + embedding baseline

Engineering register. This decides whether the prior Region B binder 'win' is REAL or an artifact.
Result: it is an ARTIFACT. The prior claim is CORRECTED. Verdict: REGION_B_BINDER_NEGATIVE.

## What was tested
The d4e0b80 Region B result claimed the fresh Qwen-geometry ContentAddressedRoleBinder 'beats both
lookups' (median 70% vs fuzzy 46%). That comparison used ONLY exact (0%) and fuzzy token-overlap
lookup and NO embedding baseline. This validation adds, on Qwen2.5-7B-Instruct 4-bit (fixed model):
- a train/test split DISJOINT BY FACT (a country->capital pair is entirely in train OR held-out test);
- a frozen embedding-retrieval baseline (sentence-transformers/all-MiniLM-L6-v2, clue<->country cosine);
- a shuffled-hidden-state ABLATION (features permuted across examples, labels kept) as a causal control;
- 12 seeds with full distributions; the four baselines on the HELD-OUT set only.

## Held-out result (cleaned data: 51 leak-free model-known facts, 16 held-out facts, 60 items, 12 seeds)
- exact (entity,attribute) lookup: 0.0%
- fuzzy token-overlap lookup:      60.0%
- MiniLM embedding cosine:         90.0%
- NEURAL BINDER:                   median 63.3% [53.3%/71.7%] std 0.054
- binder ablated (shuffled hidden states): median 56.7% [50.0%/65.0%]
- in-sample binder (logged, NEVER the result): 100.0%

A first run on the d4e0b80-faithful 59-fact set (with the pre-fix clue leakage) gave the same verdict:
binder median 59.2% vs MiniLM 83.3% on held-out, ablation 50.0%. Cleaning the clue leakage widened the
gap (MiniLM rose to 90.0%), so the negative is robust to the data-hygiene fix.

## Gates
- G_HELDOUT: held-out reported (in-sample 100% logged only, never the result).
- G_ABLATION: PASS -- ablated binder 56.7% <= chance+10%. The binder's signal
  IS causally in Qwen hidden states, but it is WEAK: only ~7pp above its own ablation.
- G_VS_EMBED: FAIL -- binder 63.3% does NOT beat MiniLM 90.0% by >= 5pp; it LOSES by
  -27pp.
- G_VS_FUZZY: context only (binder beats fuzzy, but beating fuzzy is not sufficient).
- G_SEEDS: PASS (12 seeds, full distributions).

## Independent adversarial audit (5 agents)
A multi-agent audit (binder-crippled / minilm-advantaged / split-ablation-seeds / prior-claim lenses +
a synthesis judge) was run before finalizing. Synthesis (high confidence): the NEGATIVE verdict HOLDS.
- binder_crippled: False | minilm_baseline_fair: True | verdict_holds: True
- The binder is NOT crippled: 0/800 span mis-localizations, feature<->label alignment preserved, verdict
  arithmetic reproduces, and the binder receives a SUPERSET of MiniLM's inputs plus Qwen world knowledge.
- The MiniLM baseline is FAIR: it never receives the capital, the label, or the country->capital table;
  inputs are symmetric (clue<->country-name cosine). It is a legitimately strong frozen retriever.
- The two real fairness skews (clue->own-country token leakage inflating embed; the binder's abstain
  class + wrong-bind penalty) BOTH run AGAINST the binder, so the NEGATIVE verdict is conservative.
- Data hygiene fixed per audit: make_clue now rejects token-overlap (not just exact-string) leakage of
  the country/capital name (8 leaky clues dropped). Remaining noise: a few model-generated clues are
  hallucinated or point at a different famous city; this caps the ceiling for ALL methods equally and is
  disclosed, not hidden.

## Correction of the prior d4e0b80 claim
The d4e0b80 'Region B binder win' is RETRACTED. On a proper held-out evaluation with a frozen embedding
baseline, a generic off-the-shelf MiniLM retriever beats the neural binder decisively. The binder reads
only weak, causally-real signal from Qwen's hidden states and does NOT generalize to unseen entities as
well as a pretrained embedding space. The earlier comparison against only string lookups manufactured a
false positive.

## Honest scope
A binder win HERE would have meant 'content-addressed resolution of an obscured entity from Qwen hidden
states, beating embedding retrieval', NOT 'unlocking latent model knowledge'. The binder did NOT achieve
that. MEASURED on Qwen2.5-7B 4-bit, single env. No external citation, no merge to main. This is a clean,
audited NEGATIVE result -- a valid and useful outcome.
