# Stage U Step 1 - feasibility: honest mechanics on a continuous-value substrate

Engineering register. MEASURED, deterministic. Source: `runs/stage_u/results/verdict.json`
(`scripts/certify_stage_u_arbiter.py`). Oracle = the sealed symbolic organ. NO trained
DCortexV2Model is used (none exists on disk); this characterizes the ARBITER MECHANISM and
the separability it requires, NOT a trained representation.

## The question
The symbolic organ's wrong_commit = 0 comes from EXACT integer value equality. Can the same
honest mechanics (committed / provisional / disputed + promote N=2 / retrograde M=2 / prune
K_stale=3 / reconcile) be carried by an arbiter whose value-identity test is a similarity on
CONTINUOUS 768-dim vectors instead of integer equality - and if so, what is the limit?

## What this DOES show

1. **Mechanism: the continuous arbiter reproduces the symbolic organ exactly when values are
   separable.** At rho=0 (orthogonal values), sigma=0, both value-identity functions give
   wrong_commit 0/140 and substantive op-set (RETROGRADE/PROMOTE/PRUNE) 100/100 across the 100
   L1-L5 sequences, checked against the live organ (`organ.query` + the consolidation audit log,
   not hand-written gold). The committed outcome is produced by the consolidation logic
   (hand-traced for L1 promote, L2 retrograde, L5 prune), not hardcoded.

2. **The honest property survives across a WIDE region, set by SEPARABILITY - not by the
   identity function.** The determinant is the separability MARGIN = min(same-value cosine) -
   max(different-value cosine) under observation noise. Where margin > 0 a calibrated boundary
   reaches wrong_commit = 0; where it collapses (<= 0) no threshold can separate.

   Separability margin (rows sigma, cols rho):
   ```
     sig\rho  0.00  0.30  0.50  0.70  0.85  0.95  0.99
       0.00   1.00  0.70  0.50  0.30  0.15  0.05  0.01
       0.30   0.86  0.59  0.41  0.23  0.10  0.01 -0.02
       0.80   0.48  0.29  0.17  0.05 -0.02 -0.09 -0.11
   ```
   Calibrated-cosine wrong_commit_rate is 0 wherever margin > 0 (up to rho <= 0.95 at sigma <=
   0.3), breaking only as the margin collapses (rho 0.99: 0.09 at sigma 0.3, 0.26 at sigma 0.8).
   Discretize-to-prototype is 0 everywhere except the extreme corner (rho 0.99, sigma 0.8: 0.05).

3. **Falsifiable.** Injecting a wrong value vector (white for red) commits white vs the oracle's
   red and fires wrong_commit; corrupting an L1 challenger breaks the promotion path and also
   fires. The gate is sensitive in the consolidation logic, not only on first writes.

## What this does NOT show (honest scope)

- **It is not a claim about a trained model.** No trained DCortexV2Model checkpoint exists; the
  value vectors are CONTROLLED (separability rho + noise sigma). The grids give the TARGET a
  learned representation must hit (keep same-value observations more similar than the gap to
  different values), not evidence that training reaches it. Measuring a trained model's actual
  (rho, sigma) and placing it on these grids is Stage U Step 2.
- **The first version overstated a "cosine floor at rho>0.7".** An adversarial review showed
  that was a CALIBRATION ARTIFACT of a fixed threshold theta=(1+rho)/2; with a noise-aware
  (calibrated) threshold the cosine floor moves out to rho <= 0.95, matching the margin. This
  report uses the calibrated threshold; the corrected finding is that cosine and discretize both
  track the separability margin.
- **Noise geometry caveat.** Norm-bounded noise in 768-dim has projection ~sigma/sqrt(D) on the
  ~1-dim discriminating axis, so it is benign for nearest-prototype - which is exactly why
  discretize is robust (a real high-dimensional property). This is NOT a claim that any noise is
  harmless; structured noise concentrated on the discriminating direction would stress it harder.
- The op comparison is SET-based on the substantive ops; exact op COUNTS differ from the organ's
  RECONCILE bookkeeping convention and are not certified.

## The refined conclusion (and what it means for the vision)
The honest property (wrong_commit = 0, correct promote/retrograde/prune) IS reachable on a
continuous-value substrate - the mechanism is proven against the sealed organ. The user's thesis
("the honest property comes from discreteness") is REFINED by the data: it comes from
SEPARABILITY (a positive margin between same-value and different-value distributions). Discreteness
is one way to guarantee separability; a well-separated continuous representation achieves it too,
up to ~95% inter-value cosine overlap at moderate noise. So the neural substrate CAN host the
honest mechanics - provided Stage U Step 2 trains (or constrains) the value representation to keep
that margin positive. That margin target, not a yes/no, is the deliverable of this step.

## Files
- `stage_u/neural_arbiter.py` - continuous-value committed/provisional/disputed + Pas7a ops.
- `stage_u/l_regime.py` - L1-L5 sequences + value->vector at controlled rho; symbolic oracle runner.
- `scripts/certify_stage_u_arbiter.py` - margin + calibrated wrong_commit grids + falsifiability.
- `runs/stage_u/results/verdict.json` - the measured grids.
Scope: MEASURED, controlled vectors, single machine, oracle = sealed organ; dcortex/ and steps/13
byte-identical (read-only).
