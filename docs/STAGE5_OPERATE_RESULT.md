# Stage 5 OPERATE-OVER-MEMORY - MEASURED: comparison OPERATES OVER PERSISTED MEMORY; chaining does not

Cert: scripts/certify_stage5_operate_memory.py | verdict: runs/stage5_operate/results/verdict.json
Frozen separable encoder WRITES facts into the memory banks; a NEW operation layer (the only trained
part) reads ONLY (bank tensors + query keys), never text. Held-out 14/16, 5 seeds. VERDICT STAGE_5_PARTIAL.

## The validity battery (lead with it)
| condition                                  | comparison | chaining (2-hop shuffled) |
|--------------------------------------------|------------|---------------------------|
| G_IN_MEMORY (forward reads banks+query, no text; structural) | PASS | PASS |
| bank accuracy held-out (bar 0.80)          | 1.000      | 0.372 (FAIL)              |
| G_BANK_GROUNDED: answer follows shuffled store (bar 0.90) | 1.000 (PASS) | 0.333 (FAIL) |
| BANK vs REP (bank vs input-rep)            | 1.0 vs 0.46 (bank >> rep) | 0.37 vs 0.14 |
| operates-over-memory                       | YES        | NO                        |
| G_ABSTAIN (broken chain -> abstain, bar 0.80) | -       | 0.417 (FAIL)              |

## What this proves (comparison) - the axis inversion, demonstrated at small scale
COMPARISON genuinely OPERATES OVER PERSISTED MEMORY. All four validity conditions hold:
1. The operation reads ONLY the banks + query keys - no source text, no encoder text-hidden
   (structural: the forward signature cannot receive text).
2. It is exact from the banks (1.000 held-out, 5 seeds, std 0).
3. THE PROOF: when the STORED values are shuffled across slots, the answer FOLLOWS the shuffled store
   PERFECTLY (G_BANK_GROUNDED comparison = 1.000). It computes from the persisted state, not from a
   memorized entity-to-answer mapping. This is the decisive operate-over-memory control, and it passes.
4. It is BETTER from the persisted banks than from the input representation (bank 1.0 vs rep 0.46) -
   the opposite of input-rep computation. The store carries the operation.
This is the vision's heart at small scale: the model reaches a conclusion (which is bigger) by
operating over what it persisted, with the source text absent.

## What does NOT work (lead with the negative too)
- CHAINING (2-hop pointer-follow over banks): accuracy 0.372 (fails 0.80) AND not bank-grounded
  (0.333). Following B -> (pointer stored in B's slot) -> A over the banks did not generalize. The
  bank-written value for "B same color as A" does not expose a followable address to A's slot well
  enough for the operation to chain. Chaining-over-banks remains the frontier.
- ABSTAIN on broken chains: 0.417 (fails). Tied to chaining not working.

## Honest process note (self-caught aggregation bug)
The first verdict label was STAGE_5_REP_ONLY - a BUG: the code AVERAGED grounded_compare (1.0) and
grounded_chain (0.33) into 0.667 and applied one gate. Per-facet is correct (comparison fully grounded,
chaining not). The verdict logic was fixed to per-facet gates and recomputed from the cached per-seed
data (real measured numbers, only the label changed) -> STAGE_5_PARTIAL. The risk was pre-flagged
before reading the breakdown.

## Where this leaves the vision
Operate-over-persisted-memory is DEMONSTRATED for comparison (with the full validity battery incl. the
bank-grounded proof) - the axis inversion is real at small scale for an ordering operation. Chaining
over banks is the remaining frontier. Next, inside the road: make the stored value expose a followable
pointer (so 2-hop chaining works over banks), or an iterative addressing that recovers the pointer;
multi-seed already done (5). Single architecture, templated, small synthetic - NOT generality.
