# D_Cortex — PROJECT STATUS (single source of truth)

Patent EP25216372.0 · Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
Last updated: 2026-06-20. This file is the current-state summary; per-stage detail lives in the
linked result docs. History is preserved (CHANGELOG.md, docs/PROGRESS.md) and never rewritten.

## The vision (unchanged)
An intelligence for which MEMORY is the organ of thought and language is only the access surface. The
inverted axis: input -> internalization -> structured memory -> OPERATION over memory -> language. The
model THINKS in memory; tokens are demoted to I/O. The heart is operation over memory (layer 5); the
rest serves it.

## Where the project is now (2026-06-20), in one paragraph
The MECHANISM ARC is PROVEN at small scale and validity-gated: the model internalizes facts, stores
them honestly (wrong_commit=0), and OPERATES over its PERSISTED memory with the source text absent -
both single-step (comparison) and relational (multi-hop graph traversal) - and ABSTAINS when a chain
cannot resolve (the honesty invariant extended to multi-hop). The remaining frontier is the free-text
EXTRACTION front-end: it was MEASURED to be a PRETRAINING (paraphrase-robustness) property, not a
fine-tuning-diversity property, on the small from-scratch closed-vocab substrate. So the extraction
frontier and the scale frontier MERGE into one move: port the proven mechanism arc to a pretrained base.

## Two layers of foundation (proven earlier, preserved)
- **v11 (2026-04-18, SEALED):** memory-conditioned token emission - memory as a functional layer
  separate from weights and context.
- **v15.7a / Pas 7a (2026-04-26, SEALED):** the symbolic longitudinal organ - memory self-revises at
  episode boundaries (reconcile -> prune -> retrograde -> promote), 10/10 D9 gates green, wrong_commit
  <= 0.02. This is the honest-mechanics oracle the neural work is held against.

## The integration-spine arc (this campaign) - measured, commit chain 1879c60 -> 3a3c2f4
Every step is a falsifiable gate with the dangerous direction reported first; negatives lead.

| Stage | Question | Verdict (measured) | Doc |
|-------|----------|--------------------|-----|
| Stage U | Honest mechanics on the NEURAL model's own internalized values | wrong_commit 0/140, CLEAN | STAGE_U_*.md |
| Stage I | Auto-extraction binding on multi-fact text (frozen base) | BINDING_FAIL (0.21, unsafe) | STAGE_I_RESULT.md |
| Multi-object step 1 | Is co-occurring binding recoverable from the frozen rep | BINDING_ABSENT (base retrain needed) | MULTIOBJECT_STEP1_RESULT.md |
| Multi-object step 2-3 | Train the base for separable objects; re-test | separability TRAINABLE 0.92; chaining 0.21->0.73 | MULTIOBJECT_STEP2_3_RESULT.md |
| Stage 5 | Operate over persisted memory: comparison | DEMONSTRATED + bank-grounded (1.0 / grounded 1.0) | STAGE5_OPERATE_RESULT.md |
| Stage 5b | Graph traversal via operation-side pointer recovery | REFUTED (pointer not recoverable) | STAGE5B_GRAPH_RESULT.md |
| Stage 5c | Encoder pointer-write (learned pointer representation) | REFUTED (held-out 0.50, the deep finding) | STAGE5C_POINTER_RESULT.md |
| Stage 5d | Structural addressing (pointer = copy of target content-key) | GRAPH TRAVERSAL PROVEN (struct 1.0 vs learned 0.45; chaining 1.0, grounded 0.949) | STAGE5D_STRUCTURAL_RESULT.md |
| Stage 5e | Honest traversal (abstain on broken chains) | HONEST_TRAVERSAL (abstain 0.858, over 0.022, dual gate) | STAGE5E_HONEST_RESULT.md |
| Stage 6 | Free-text extraction from varied phrasing | SUBSTRATE_LIMITED (value 0.56, wrong-bind 0.38) | STAGE6_EXTRACTION_RESULT.md |
| Stage 7 | Substrate fine-tune for phrasing robustness + arc re-verify | PHRASING_REFUTED; arc FULLY PRESERVED (Stage U 0/140, traversal 1.0/0.994, abstain preserved) | STAGE7_SUBSTRATE_RESULT.md |
| Stage 8 | Scale phrasing diversity; structurally-distinct held-out | REFUTED_AT_SCALE (value 0.337 near chance) -> paraphrase-robustness is a PRETRAINING property | STAGE8_PHRASING_SCALE_RESULT.md |

## The deep finding (what the arc taught)
- ONE object is clean everywhere (storage, honesty, internalization, canonical, single-step operation).
- MULTI-object separation/operation was the root wall; it is TRAINABLE into the base (Step 2) and
  unlocks single-step operation (Stage 5) and graph traversal via STRUCTURAL addressing (Stage 5d:
  store the relational pointer as a COPY of the target's content-key, reusing content-addressing which
  already generalizes - not a learned pointer representation, which was refuted in 5c).
- VALUE-identity separability GENERALIZES across entities (a content-slot property the substrate holds);
  PHRASING-invariance (paraphrase-robustness over surface form) does NOT generalize on a from-scratch
  closed-vocab substrate even with scaled diversity - it is a pretraining / language-understanding
  property (Stage 8).
- HONESTY transfers: wrong_commit=0 holds on single facts, on canonical writes, and (extended) on
  multi-hop traversal (Stage 5e); the abstain mechanism even transfers to the extraction front-end
  (broken-pointer abstain 0.96 in Stage 6) - only BINDING is substrate/pretraining-limited.

## What is PROVEN (the durable asset)
The operate-over-persisted-memory mechanism, honest and validity-gated: single-step operation
(comparison, bank-grounded), relational graph traversal (structural addressing, chain-grounded), honest
abstain (dual gate, anti-collapse), single-fact honesty preserved throughout (Stage U 0/140), and
robust to a gentle substrate fine-tune (Stage 7/8 re-verification). All with the source text ABSENT
from the operation (G_IN_MEMORY structural).

## The frontier (next move, localized)
PORT the proven mechanism arc (Stage 5 -> 5e) to a PRETRAINED base, where paraphrase-robust extraction
is tractable. Stage 8 showed free-text extraction-binding is coupled to pretraining, so the EXTRACTION
frontier and the SCALE frontier are ONE move, not two. The mechanism is proven; the path to autonomy is
the pretrained base.

## Validity discipline (how every claim here was earned)
Falsifiable pre-declared gates; lead with the dangerous direction (wrong-binding, cross-binding) not
raw recovery; bank-grounded / chain-grounded controls (shuffle the store -> the answer must follow);
structural-vs-learned A/B; multi-distractor (never single-option); double held-out (entities AND
phrasings); >=5 seeds with full distributions; the operation never sees text (structural assert).
Negatives are first-class results that localize the next move. dcortex/ and steps/ are sealed/read-only;
trained weights (*.pt) are gitignored, never committed.

## Scope honesty
Single architecture, closed vocab, small synthetic, held-out splits - a MECHANISM diagnosis, NOT
generality or scale. The claims are about reachability of the mechanism, proven cleanly small; scaling
and open-domain are the pretrained-base frontier.
