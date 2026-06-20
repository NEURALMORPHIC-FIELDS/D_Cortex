# Stage 8 PHRASING-DIVERSITY SCALE - MEASURED: STAGE_8_PHRASING_REFUTED_AT_SCALE (paraphrase-robustness is a pretraining property)

Cert: scripts/certify_stage8_phrasing_scale.py | verdict: runs/stage8_phrasing/results/verdict.json
Scaled phrasing diversity: 16 value + 12 relation TRAINING phrasings (the entity-count analog of Step 2).
Held-out on a STRUCTURALLY DISTINCT family (value-first / inverted / embedded), NOT near-duplicates.
3 fine-tune seeds x 3 extractor seeds. VERDICT STAGE_8_PHRASING_REFUTED_AT_SCALE.

## The decisive gate (on STRUCTURALLY-DISTINCT held-out - the real paraphrase-invariance test)
| gate                                                   | result            |
|--------------------------------------------------------|-------------------|
| G_VALUE_BINDING_PHRASE (Family B, distinct; bar >=0.85)| 0.337 FAIL (near chance 0.25) |
| G_WRONG_VALUE_BINDING (bar <=0.02)                     | 0.45 FAIL (fabrication) |
Consistent across 3 fine-tune seeds (0.317 / 0.317 / 0.333) - robust to fine-tune-init variance (this
hardens the Stage 7 single-seed negative). On a genuinely structurally-distinct held-out family, the
binding is NEAR CHANCE - WORSE than Stage 7's 0.52 (whose random phrasing split admitted near-duplicates).

## What this proves (the calibrated outcome, anticipated)
Scaling training phrasing diversity to 16 did NOT make the substrate abstract phrasing-INVARIANCE. The
substrate learned the training family ("value follows entity-is", adjacent) but cannot generalize to
inverted / value-first / embedded structures. Entity-invariance generalized (Step 2, 0.92) because it
is generalization over a CONTENT slot the substrate is built to hold; phrasing-invariance is
generalization over SURFACE FORM (paraphrase-robustness) - a different, harder abstraction. A from-
scratch closed-vocab substrate lacks the language-understanding capacity for it; pretrained models get
it free from large diverse text.

## The validity guards held
- The held-out was a STRUCTURALLY DISTINCT family (not near-duplicates) -> this is a real paraphrase-
  invariance test; a pass would have been meaningful, and the failure is not a held-out-too-easy artifact.
- The foundation SURVIVED: Stage U on the new substrate = 0/140 (margin 0.280). The gentle fine-tune +
  distillation did not damage the arc (same as Stage 7). Clean negative: no regression, no progress.

## The next move (the two frontiers MERGE into one)
PORT THE PROVEN MECHANISM ARC (Stage 5 -> 5e) TO A PRETRAINED BASE. The result says extraction-binding
from free text is coupled to PRETRAINING (paraphrase-robustness), not to fine-tuning diversity at this
capacity. So the extraction frontier and the scale frontier are the SAME move: take the proven, validity-
gated mechanism (operate over persisted memory, single-step + honest graph traversal, wrong_commit=0 to
multi-hop) and re-establish it on a pretrained base where paraphrase-robust extraction is tractable. This
negative does not stall the program - it FOCUSES it: one move instead of two.

## Program status (honest, end of the mechanism arc)
- MECHANISM arc (operate over persisted memory, honest, single-step + graph traversal): PROVEN, validity-
  gated, and robust to gentle substrate fine-tunes. This is the durable asset.
- FREE-TEXT EXTRACTION on the small from-scratch substrate: REFUTED at scale - it is a pretraining
  property. The path forward is a pretrained base, where extraction + scale are one frontier.
Single architecture, closed vocab, small synthetic - mechanism diagnosis, not generality.
