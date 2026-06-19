# Multi-object road, STEP 1 - MEASURED: BINDING_ABSENT_IN_FROZEN_REP (base retrain is mandatory)

Cert: scripts/certify_multiobject_readout.py | verdict: runs/multiobject_readout/results/verdict.json
Frozen ckpt_multiattr | multi-entity same-attr distinct-value regime | held-out 10/30 entities | 5 seeds.

## The question
Stage I showed a LINEAR probe at the entity position cannot recover the per-entity value in multi-fact
text. Is the binding PRESENT-but-nonlinear (close: a readout/light training unlocks it) or ABSENT from
the frozen rep (far: the base must be retrained)? We climbed a readout ladder on the SAME frozen rep.

## Result (held-out wrong_binding median, 5 seeds)
| Readout              | held-out | in-sample | reads |
|---------------------|----------|-----------|-------|
| linear              | 0.1825   | 0.096     | the Stage I baseline |
| mlp (1 hidden, GELU)| 0.1400   | 0.000     | fits train PERFECTLY, fails held-out |
| attention (4-head)  | 0.1975   | 0.069     | fits train (0.93), fails held-out |

Bar for PRESENT was <= 0.05; best nonlinear = 0.14 -> DIRECTION = BINDING_ABSENT_IN_FROZEN_REP.

## Why this is trustworthy (a confound was caught and fixed)
The first attention attempt FAILED to fit even in-sample (wb 0.19) - an undertrained readout, not
evidence about the rep. The attention readout was strengthened (multi-head + pre-norm + residual MLP,
2x steps); it then FIT in-sample (0.069) and STILL failed held-out (0.20). So the held-out failure
indicts the representation, not the readout.

## The decisive signature: a generalization gap, not a capacity gap
The MLP fits train PERFECTLY (in-sample wrong_binding = 0.000) but fails held-out (0.14). The binding
is memorizable per-entity, but there is NO generalizable separable-object structure in the frozen rep:
a readout trained on 20 entities cannot read co-occurring bindings for 10 unseen entities. Adding
readout depth would memorize train better, NOT generalize better. Only changing the REPRESENTATION
(training the base) can add the structure.

## Caveat (stated, not oversold)
Tested readouts: linear, 1-hidden MLP, 4-head single-layer attention. A much deeper readout might
extract more in-sample, but the failure mode here is generalization, not capacity - so deeper readouts
do not address it. The honest conclusion is specific: no MODEST readout recovers GENERALIZABLE
co-occurring binding from this frozen rep.

## DIRECTION (one road, confirmed)
FAR. Step 2 (retrain the base to MAINTAIN separable co-occurring objects) is MANDATORY, not optional -
exactly the gap predicted: the base was trained single-fact-per-encode, never multi-fact. Step 3 then
re-tests binding (Stage I) and chaining (Stage C) on the retrained base; same root -> both should
unblock together if the structure is learned. v1 single-fact remains the safety net if the root resists.
