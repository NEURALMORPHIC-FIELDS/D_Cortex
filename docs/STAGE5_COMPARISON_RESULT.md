# Stage 5 (operate-over-memory), first piece COMPARISON - MEASURED: REACHABLE (gate cleared)

Cert: scripts/certify_stage5_comparison.py | verdict: runs/stage5_comparison/results/verdict.json
A/B frozen bases, explicit operation module over two co-occurring object reps, held-out 10/30, 5 seeds.

## The question
Stage C C2 (comparison) was at chance. Is that because (a) the value space lacks ordinal structure,
(b) comparison is unreachable, or (c) the OPERATION module was simply never built (the sealed decoder
sums slots and cannot compare)? Build the explicit operation module and measure.

## Result (held-out median, 5 seeds)
| arm                         | ordinality (rank, 4-way) | comparison (which bigger) |
|-----------------------------|--------------------------|---------------------------|
| baseline ckpt_multiattr     | 0.754                    | 0.819                     |
| separable ckpt_multiobject  | 0.877                    | 0.883                     |
Gate for comparison: held-out >= 0.80 (chance 0.5). BOTH arms clear it; separable 0.883.

## Verdict: COMPARISON_REACHABLE
An explicit operation module reading two separable object representations SOLVES comparison held-out
(0.883), clearing the 0.80 gate. Stage C's C2-at-chance was the MISSING OPERATION MODULE, not a
representation that lacks ordinality (ordinality 0.88 held-out) nor an unreachable operation. This is
exactly the layer Stage C's diagnosis said must be BUILT - the vision's layer 5 (operate-over-memory)
is reachable for comparison.

## Honest nuance (lead with it, do not oversell)
- Comparison clears the gate on the BASELINE base too (0.819), so for comparison the OPERATION MODULE
  is the decisive piece; separability is a modest boost (+0.06, below the 0.10 enabler threshold), NOT
  the gate. (Contrast: for chaining, the separable init was the large lever, 0.21->0.73.)
- The module reads the ENCODER per-entity object reps, not a decode-from-banks read. Faithful in
  spirit (two separable objects -> operation); a bank-read version is the follow-up.
- Single architecture, 4 sizes, small synthetic. Symmetric pairs (both orders) remove positional bias;
  held-out entities make it a genuine ordinal-transfer test, not memorization.

## Where this leaves the road
Layer 5 is buildable on the substrate: comparison (an ordering operation) is reachable by an explicit
operation module, gate cleared, generalizing held-out. Next, inside the road: (1) the faithful
bank-read operation (operate over the written memory slots, not the encoder rep); (2) robust chaining
to the gate via the operation layer (C1 was 0.73, just under 0.80); (3) multi-seed to pin numbers.
