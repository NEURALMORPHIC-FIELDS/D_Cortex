# Stage 5c ENCODER POINTER-WRITE - MEASURED: REFUTED (relational pointers do not generalize across entities)

Cert: scripts/certify_stage5c_pointer_write.py | verdict: runs/stage5c_pointer/results/verdict.json
Encoder relational-writing fine-tuned (encoder.blocks, contrastive pointer loss + distillation
preservation); Stage 5b operation reused. Held-out 14/16, 5 seeds. VERDICT STAGE_5_POINTER_REFUTED.

## STEP 0 diagnostic (frozen encoder) - the case
Recover A's key from the FROZEN B-slot value: in-sample 0.985, held-out 0.355 -> ENTITY-SPECIFIC. The
pointer info IS present (perfect in-sample) but encoded per-entity; it does not generalize. The encoder
fix is necessary and must RESTRUCTURE the pointer into a generalizable address.

## Result (5 seeds) - lead with the negative
| gate                                            | result            |
|-------------------------------------------------|-------------------|
| G_POINTER_RECOVERY held-out (bar 0.90)          | 0.500 FAIL (in-sample 1.000) |
| G_CHAINING_BANK (2-hop shuffled; bar 0.80)      | 0.382 FAIL        |
| G_CHAIN_GROUNDED (re-point follows; bar 0.90)   | 0.350 FAIL        |
| G_ABSTAIN (bar 0.80)                            | 0.381 FAIL        |
| G_COMPARISON_PRESERVED (bar 0.80)               | 1.000 PASS        |
| G_SINGLE_FACT_PRESERVED (value cosine to frozen)| 1.0000 PASS       |
Even a dedicated encoder pointer-write fine-tune (contrastive supervision to the target key) does NOT
make A's address recoverable on HELD-OUT entities: in-sample recovery is perfect (1.0), held-out is
0.50 (barely above the 3-candidate chance 0.33). The entity-specific gap persists. Graph traversal
stays at ~0.38 (grounded ~0.35). The distillation kept non-relational value-writing perfectly intact
(cosine 1.0) and comparison perfect (1.0) - so the failure is specific to the relational pointer, not
collateral damage.

## THE DEEP FINDING (the real result of the Stage 5b/5c arc)
A precise boundary in the architecture at this setup:
- VALUE-IDENTITY separability GENERALIZES across entities (Step 2: 0.92 held-out). "What value does
  this object hold" transfers to unseen entities.
- RELATIONAL-POINTER address does NOT generalize (here: in-sample 1.0, held-out 0.50, even with a
  dedicated pointer-write). "Which other object does this point to, as a followable address" can be
  memorized per train-entity but does not transfer to unseen entities.
So the architecture can store and OPERATE on per-object VALUES over persisted memory (comparison
demonstrated + bank-grounded, Stage 5), but it cannot store a GENERALIZABLE relational POINTER to make
memory a navigable GRAPH. Multi-hop graph traversal is blocked by the encoder's inability to write a
generalizable followable pointer - not by the operation (Stage 5b) and not by the value path.

## What this means for the vision
- Operate-over-PERSISTED-memory for a per-object operation (comparison): DEMONSTRATED + grounded
  (Stage 5). Single-step thinking-in-memory is real at small scale.
- Graph traversal (multi-hop, follow stored relational pointers): REFUTED at this setup. The pointer is
  not generalizable. Making memory a navigable graph needs a fundamentally different RELATIONAL-
  ADDRESSING mechanism than value separability - a deeper architectural change (e.g. a dedicated
  relational key space, a learned address-binding objective, or an architecture where the target's key
  is copied structurally rather than recovered from content). That is the precise next frontier.

## Caveat (honest)
Single architecture, templated, 6 colors / small synthetic, one pointer-write recipe (encoder.blocks +
contrastive + distill). A different relational-writing mechanism might generalize; this refutes THIS
recipe, strongly, and localizes the limit to relational-pointer generalization.
