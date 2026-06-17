# Stage U - Unification: honest mechanics on the NEURAL (continuous-value) substrate

Autonomous campaign (architectural autonomy granted 2026-06-17). The single rule: respect the
vision direction (memory as the organ of thought; honest, falsifiable mechanics). Anti-confabulation
in full force: no unproven success, failures reported first, gates falsifiable.

## The vision-level goal of Stage U
Port the symbolic organ's honest mechanics (committed / provisional / disputed arbitration +
promote / retrograde / prune / reconcile consolidation) onto the NEURAL banks of DCortexV2Model,
where a value is a CONTINUOUS 768-dim vector, not a discrete index. Done (falsifiable): on the
F1-F5 / L1-L5 regime the neural arbiter reaches wrong_commit = 0 and the correct op-sequence, i.e.
the symbolic gates pass on the neural substrate.

## The exact open problem (grounded in the code)
The symbolic organ's wrong_commit = 0 comes from EXACT integer equality:
- conflict routing is `slot.value_idx != value_idx` (steps/13 code.py:14321),
- promotion counts distinct episode_ids where the SAME value_idx appeared (N_promote = 2),
- scoring is `pred_value == target_value_idx` (code.py:11907).
On the neural banks "same value vs different" is a THRESHOLD on 768-dim vectors (theta_conflict),
which has a false-match / false-miss floor; "count episodes supporting value V" first needs the
vectors CLUSTERED into value classes. So the honest property may or may not survive on continuous
values - that is exactly what Stage U must MEASURE, not assume.

## Crux decisions (autonomous, documented, falsifiable)
1. Checkpoint-or-train: there is NO trained DCortexV2Model checkpoint on disk (only small
   binder/adapter `*_head.pt`). So Step 1 does NOT depend on a trained model. It isolates the
   ARBITER MECHANISM from REPRESENTATION QUALITY via a controlled value-separability parameter rho
   (the pairwise cosine between distinct symbolic values). This answers "is wrong_commit = 0
   reachable on continuous values, and at what separability margin does it break" without training.
   Training the full model to land its learned values in the honest region is a later, separate step.
2. Value-identity function (the validity-critical knob): pluggable.
   - PRIMARY = cosine threshold on the raw value vectors (faithful to the vision: operate on
     continuous values).
   - SECONDARY = discretize each value to its nearest prototype, then exact-equal (the "regain
     exact equality" option; it is a discrete code, reported as such).
3. dcortex/ is now editable, but Step 1 keeps it UNTOUCHED: the arbiter is a NEW, bank-agnostic
   module operating on value vectors. It wires INTO DCortexV2Model only after the mechanism is proven.
4. Oracle = the symbolic organ (Pas7a 10/10 on L1-L5). The L1-L5 gold op-sequence + committed value
   per level are the reference the neural arbiter must reproduce.
5. Gate (falsifiable): wrong_commit = committed-value (decoded to the nearest value prototype, mapped
   to a symbolic id) != gold committed id; plus op-sequence match vs the L-level gold. Falsifiability:
   injecting a corrupted value vector must fire wrong_commit.

## L1-L5 regime (the cross-episode consolidation families, 20 sequences each, 100 total)
- L1 promote_cycle:   committed red; 2 challenger-blue episodes; distractor -> RECONCILE 1, RETROGRADE 1, PROMOTE 1 (ends committed = blue)
- L2 retrograde_only: committed red; 2 challenger-blue episodes               -> RECONCILE 1, RETROGRADE 1 (ends with red demoted, no new commit)
- L3 completion:      same entity gets color, then size, then location         -> 0 ops (3 distinct attributes, no conflict)
- L4 no_inflation:    intra-ep1 conflict (red, red, blue); 3 distractor eps     -> RECONCILE 1 (red stays, single observation does not inflate)
- L5 stale_prune:     conflict ep1 (red, blue); 3 distractor eps to K_stale = 3 -> RECONCILE 1, PRUNE 2

## Step 1 deliverable
- stage_u/neural_arbiter.py: NeuralCommitArbiter (continuous-value committed/provisional/disputed +
  promote N=2 / retrograde M=2 / prune K_stale=3 / reconcile), pluggable value-identity.
- stage_u/l_regime.py: generate the 100 L1-L5 sequences; map symbolic values to 768-dim vectors at a
  controlled pairwise cosine rho.
- scripts/certify_stage_u_arbiter.py: run the arbiter over the regime across a rho / theta sweep;
  report wrong_commit(rho) and op-sequence match; falsifiability check; the discretized control.

## Honest scope of Step 1
This measures the ARBITER MECHANISM + the separability requirement on CONTROLLED value vectors. It
does NOT yet use a trained DCortexV2Model's learned values (none exists). The wrong_commit(rho) curve
gives the TARGET margin a trained representation must achieve - it is the feasibility map for the
vision on the neural substrate, not a claim that the trained model is already honest.
