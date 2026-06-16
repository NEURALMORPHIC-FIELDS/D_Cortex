# Capable-model professional integration: two regions, real plausible-hallucination veto, valid binder test

> CORRECTION (region-b-validation): the Region B binder 'win' below is RETRACTED. A held-out
> evaluation with a frozen MiniLM embedding baseline + shuffled-hidden-state ablation shows the
> binder LOSES to off-the-shelf embedding retrieval (binder ~63% vs MiniLM ~90% held-out). The
> Region A enforcement results stand; only the Region B binder-differentiator claim is overturned.
> See docs/REGION_B_VALIDATION_REPORT.md.


Engineering register. Claim status governs every line. MEASURED, single environment.

## Base model (fixed, not chosen by the assistant)
Qwen/Qwen2.5-7B-Instruct in 4bit-nf4 (bitsandbytes NF4, double quant, bf16 compute), hidden dim 3584,
RTX 5080 Blackwell sm_120, torch 2.11 cu128. The model id was specified, only verified. A fresh
ContentAddressedRoleBinder was reinitialized on this model's geometry (layer -1 hidden states); no gpt2
weights were reused. This corrects the prior run, where an unspecified 'largest model that fits' led to
gpt2-large and invalidated the binder test.

## Why two regions
REGION A (WIPO IPC pack, 192 codes): the model LACKS these facts, so it tests the enforcement /
limitation property. REGION B (probe-filtered model-known facts, both lookups made to fail): the only
place the neural binder differentiator is even testable, because a content-addressed binder can only
ground what the model's hidden states already encode.

## REGION A - enforcement on facts the model lacks
- RAW (no D_Cortex): hallucination 96.9%, grounded only 4%. The
  capable model answers IPC-title questions with confident, well-formed, WRONG titles (e.g. it titles IPC
  A23 'Agriculture; Forestry...' when A23 is foodstuffs, and A42 'Hand-operated articles' when A42 is
  headwear). This is the PLAUSIBLE-hallucination case gpt2 could not produce.
- CONTROLLED (under D_Cortex): hallucination 1.5%, recall 98%.
- RAW vs CONTROLLED delta +95.5% -- LABELLED FLOOR-DEPENDENT (depends on the raw
  model's rate), NOT a performance metric.
- All 50 committed answers were overridden: the model's unconstrained slot emission differed from the
  committed title every time; the constraint pinned each to the grounded value.

## G_VETO_REAL - a real, captured plausible hallucination, force-abstained
Qwen was asked for an uncovered IPC code and confidently fabricated the title:
  "General Methods or Articles Relating Therein"
Routed through the control layer, the answer was force-abstained and did NOT leak that fabricated title.
This is the genuine plausible-hallucination veto, now testable with a capable model.

## REGION B - the binder-favorable regime (the meaningful positive)
- built from 91 model-known country->capital facts (probe-kept: model answered them correctly
  RAW), 59 usable model-generated clues, 50 test items.
- regime VALID by pre-declared precondition: fuzzy token-overlap lookup 46% < 60%, so
  the clues genuinely defeat string lookup.
- fresh binder on Qwen/Qwen2.5-7B-Instruct layer -1 hidden states: median 70% [64%/72%] std 0.029
- vs exact (entity,attribute) lookup 0% and FAIR fuzzy lookup 46%:
  binder beats BOTH (median 70%, margin vs fuzzy +24.0pp).
- This is the result the corrected methodology produces: when the model encodes the facts and both
  lookups fail, the neural binder genuinely adds value. It is the inverse of the prior gpt2 negative,
  and confirms the lesson: the binder must be tested on a model whose representations encode the facts.

## Reconcile (hallucination vs leak)
- hallucination = any ungrounded factual assertion reaching OUTPUT (wrong value on covered, fabrication
  on uncovered, confident answer on out-of-domain); measured at output.
- leak = a committed-SLOT bypass: control returns grounded=True with a non-committed value (verifier
  bypassed); measured on the adversarial set.
- Under D_Cortex both are ~0 here (uncovered/ood abstained; verifier holds): different denominators, not
  contradictory.

## Gates (frozen) - all PASS
- [PASS] G_RAW
- [PASS] G_CONTROLLED
- [PASS] G_RECALL
- [PASS] G_ABSTAIN
- [PASS] G_FORBIDDEN
- [PASS] G_VETO_SYNTH
- [PASS] G_VETO_REAL
- [PASS] G_NOBYPASS
- [PASS] G_BINDER
- [PASS] G_TRACE

## Claim separation
- Region A: the model is inert on IPC facts; grounding is delivered by deterministic lookup + constrained
  decoding + the verifier veto. The value is enforced grounding + a fluent interface, NOT knowledge
  expansion of the model.
- Region B: the neural binder is the differentiator and, on a valid regime, beats both lookups. This is
  attributed to the binder, measured over 5 seeds.

## Claim status and external-citation constraint
VERDICT: D_CORTEX_CAPABLE_PROFESSIONAL_PASS. MEASURED on Qwen/Qwen2.5-7B-Instruct 4-bit NF4, single machine/env, single organization.
This is the capable-model run the prior report required before any external citation. It still MUST NOT
be cited externally (patent / EESR / regulatory / marketing) until multi-hardware reproduction AND
independent replication are done, and the caveats here travel with the citation. Stays on the feature
branch; not merged into main. A low controlled hallucination rate and a valid binder win here are strong
MEASURED results, not a deployment guarantee.
