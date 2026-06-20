# Stage 5e HONEST TRAVERSAL - MEASURED: STAGE_5_HONEST_TRAVERSAL (wrong_commit=0 extended to multi-hop)

Cert: scripts/certify_stage5e_honest_traversal.py | verdict: runs/stage5e_honest/results/verdict.json
FROZEN encoder; structural traversal (5d) + a confT-driven, detection-based abstain. Balanced
answerable/broken (138/134 held-out), 5 seeds. VERDICT STAGE_5_HONEST_TRAVERSAL (all gates pass).

## The DUAL abstain gate (lead with it - a high abstain rate alone is meaningless)
| gate                                                  | result            |
|-------------------------------------------------------|-------------------|
| G_ABSTAIN_BROKEN (broken chain -> abstain; bar >=0.80)| 0.858 PASS        |
| G_NO_OVER_ABSTAIN (answerable -> NOT abstain; bar <=0.10) | 0.022 PASS     |
The model abstains on broken chains AND answers when it can - no abstain-collapse (the degenerate
"always abstain" that gamed an earlier Stage C run). Abstain is DETECTION-BASED: tied to confT, the
retrieval confidence at the followed content-key. A non-stored target gives low confT (~0.15) vs a
stored target (~0.98) - a strongly separable retrieval signal that generalizes held-out (it is a
retrieval property, not entity identity).

## Preservation (nothing proven regressed)
| gate                                       | result            |
|--------------------------------------------|-------------------|
| G_CHAINING_PRESERVED (answerable acc >=0.95, grounded >=0.90) | 0.971 / 0.964 PASS |
| G_COMPARISON_PRESERVED                     | 1.000 PASS        |
| G_POINTER_GENERALIZES (structural)         | 1.000 PASS        |
| G_SINGLE_FACT_PRESERVED (frozen encoder)   | 0/140 PASS        |
| G_IN_MEMORY (reads banks + pointer bank + query, no text) | PASS (structural) |

## What this closes
The honesty invariant - the signature property of D_Cortex (never assert what it cannot support;
wrong_commit=0) - now extends to MULTI-HOP traversal. The navigable graph is HONEST: it traverses
proven relational pointers AND refuses broken chains. Combined with the arc:
- single-step operation over persisted memory (comparison, bank-grounded) - Stage 5
- multi-hop graph traversal (structural addressing) - Stage 5d
- honest abstain on broken traversal (dual gate) - Stage 5e
the thinking-in-memory loop is closed HONESTLY at small scale.

## Honest boundary (what is NOT yet covered)
- TARGET IDENTIFICATION is template-given. This closes honest reasoning over IDENTIFIED targets;
  building the graph from FREE TEXT (identifying entities/relations) is the separate extraction problem
  (Stage I refuted multi-fact binding) - the next real campaign.
- Single architecture, 6 colors / small synthetic, held-out entities - mechanism demonstration, NOT
  generality or scale.

## Where the road goes next (ordered)
1. DONE: honest traversal (this).  2. NEXT: free-text extraction (build the graph from text).
3. THEN: scale beyond the small synthetic domain.
