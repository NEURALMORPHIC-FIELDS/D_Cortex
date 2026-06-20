# Stage 5b GRAPH TRAVERSAL OVER MEMORY - MEASURED: REFUTED at this setup (recovery-from-frozen-value insufficient)

Cert: scripts/certify_stage5b_graph_traversal.py | verdict: runs/stage5b_graph/results/verdict.json
Frozen separable encoder writes facts to banks; operation layer (only trained part) with a dedicated
pointer-recovery head (auxiliary-supervised: recover(B_value) -> target slot key) + content-chained
read. Held-out 14/16, 5 seeds. VERDICT STAGE_5_CHAIN_REFUTED.

## LEAD WITH THE NEGATIVE
2-hop chaining over the banks is NOT achieved, and the answer does NOT follow the re-pointed graph.
| gate                                                 | result            |
|------------------------------------------------------|-------------------|
| G_IN_MEMORY (forward reads banks+query, no text)     | PASS (structural) |
| G_CHAIN_GROUNDED (re-point B->C, answer must follow; bar 0.90) | 0.255 FAIL |
| G_CHAINING_BANK (2-hop shuffled, banks only; bar 0.80) | 0.316 FAIL      |
| G_ABSTAIN (broken pointer -> abstain; bar 0.80)      | 0.381 FAIL        |
| G_COMPARISON_PRESERVED (bar 0.80)                    | 1.000 PASS        |
Chaining held-out is ~0.32 across 5 seeds (std small), grounded ~0.25 (near chance). Comparison stays
perfect (1.0) - no regression.

## What the negative means (precise)
The engineering choice was to try the CHEAPER coupled fix first: a pointer-recovery head IN THE
OPERATION LAYER (auxiliary-supervised to the target key), reading the FROZEN encoder's relational
value, leaving the encoder unchanged. This is REFUTED: the frozen bank value for "B same-color-as A"
does not expose a recoverable, held-out-GENERALIZABLE address to A's slot, even with direct supervised
recovery. The operation layer trains fine (comparison 1.0), but the recovery/traversal path does not
generalize.

Therefore the OTHER coupled fix is the indicated next step: EXTEND THE ENCODER'S RELATIONAL WRITING so
B's slot explicitly stores A's addressing key as a first-class, recoverable pointer (a small fine-tune
of the relational-writing, analogous to the Step-2 separability fine-tune), then re-run this cert. The
deferred encoder-side fix is not optional; the operation-side recovery alone is insufficient.

## Caveat (honest, not oversold)
Undiagnosed here: whether the recovery fails IN-SAMPLE too (the value->target-key mapping is
fundamentally not learnable) or only held-out (a generalization gap). Either way, recovery-from-frozen-
value does not yield graph traversal at this setup. A quick in-sample recovery-accuracy probe + the
encoder relational-writing extension are the next steps. Single architecture, templated, small synthetic.

## Where this leaves operate-over-memory
- SINGLE-STEP operation over persisted banks: DEMONSTRATED and bank-grounded (Stage 5: comparison acc
  1.0, grounded 1.0, bank >> rep). The axis inversion is real for a per-object operation.
- GRAPH TRAVERSAL (multi-hop, follow stored pointers): NOT yet - the relational pointer is not
  recoverably stored in the frozen banks. This is the precise remaining frontier, and the fix is on the
  ENCODER (pointer-write), not the operation layer.
