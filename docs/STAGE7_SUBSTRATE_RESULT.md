# Stage 7 SUBSTRATE FINE-TUNE - MEASURED: STAGE_7_PHRASING_REFUTED (arc preserved; extraction not improved)

Fine-tune: scripts/train_substrate_phrasing.py -> runs/multiobject/ckpt_multiobject_phrase.pt (a COPY;
ckpt_multiobject NOT overwritten). Arc re-verified by re-running the proven certs on the new substrate.
Verdict assembled in runs/stage7_substrate/results/verdict.json. VERDICT STAGE_7_PHRASING_REFUTED.

## The two things that decide it together (both measured)
### 1. Extraction on the new substrate - DID NOT IMPROVE (the decisive question)
| gate (re-test on new substrate, double held-out)     | result            |
|------------------------------------------------------|-------------------|
| G_VALUE_BINDING_PHRASE (bar >=0.85)                  | 0.52 FAIL (was 0.557 on original -> NO improvement) |
| G_WRONG_VALUE_BINDING (bar <=0.02)                   | 0.42 FAIL (fabrication) |
| G_RELATION_BINDING (bar >=0.75)                      | 0.467 FAIL        |
| G_RELATION_DIRECTION_WRONG (bar <=0.05)              | 0.533 FAIL        |
Fine-tuning on varied phrasing did NOT make extraction-binding generalize to HELD-OUT phrasings. The
fine-tune fit the training phrasings (loss 0.004) but did not abstract phrasing-invariant binding -
memorization, not generalization.

### 2. The proven arc on the new substrate - FULLY PRESERVED (the validity guard)
| arc gate (re-run on new substrate)                   | result            |
|------------------------------------------------------|-------------------|
| G_STAGE_U_PRESERVED (wrong_commit 0/140)             | 0/140, margin 0.282 PASS |
| G_COMPARISON_PRESERVED (>=0.80)                      | 0.881 PASS        |
| G_TRAVERSAL_PRESERVED (chaining 1.0, grounded 0.994) | PASS              |
| G_ABSTAIN_PRESERVED (broken 0.836, over 0.007)       | PASS (STAGE_5_HONEST_TRAVERSAL re-passes) |
| templated bank-value preservation (cosine to frozen) | 1.0000            |
The gentle fine-tune + distillation kept the ENTIRE proven arc intact - the foundation was NOT damaged.
No disguised step-back: the substrate that does (no better) extraction still carries the whole proven
mechanism arc, re-verified.

## The finding (honest, with the confound named)
PHRASING_REFUTED AT THIS DIVERSITY SCALE. Step 2 made binding generalize over ENTITIES with 14+ training
entities; here the fine-tune had only ~4 training phrasings - too little phrasing diversity to abstract
phrasing-invariance. So this refutes "this fine-tune at THIS phrasing-diversity", not "phrasing-
robustness is fundamentally impossible". The validity guard (mandatory arc re-verification) did exactly
its job: it confirmed NO regression on the proven arc AND NO progress on extraction - a clean, safe-to-
iterate negative.

## The next move (clear, scoped)
Scale up PHRASING DIVERSITY in the fine-tune - many more distinct phrasings (the entity-count analog of
Step 2), then re-run this exact pipeline. The arc re-verification confirms the fine-tune is SAFE to
iterate (it does not break the foundation), so scaling phrasing diversity is a low-risk next experiment.
If even high phrasing-diversity fails, varied-phrasing binding needs a different mechanism than the
Step-2 move. Until then: the mechanism arc (Stage 5 -> 5e) stays the proven asset; the free-text
extraction front-end remains the open frontier, now localized to "needs more phrasing diversity".

## Program status (honest)
- MECHANISM arc (operate over persisted memory, single-step + honest graph traversal): PROVEN, and now
  RE-CONFIRMED to survive a substrate fine-tune (it is robust to gentle foundation changes).
- EXTRACTION front-end from free text: NOT closed. This fine-tune did not crack it; the next lever is
  phrasing diversity. Single architecture, closed vocab, small synthetic - mechanism diagnosis.
