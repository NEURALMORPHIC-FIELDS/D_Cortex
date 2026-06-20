# Multi-object road, STEPS 2-3 - MEASURED: separability is TRAINABLE; it PARTIALLY unblocks chaining

Step 2 cert: scripts/train_multiobject.py -> runs/multiobject/ckpt_multiobject.pt (a COPY; ckpt_multiattr
NOT overwritten). Step 3 re-certs: certify_multiobject_readout.py, certify_stage_i_extraction.py,
certify_stage_u_unification.py, certify_stage_c_reasoning.py --init-ckpt. All on RTX 5080.

## STEP 2 - train the base to maintain separable co-occurring objects
Fine-tuned ONLY the encoder contextual path (encoder.emb_norm + 4 encoder blocks + encoder.final_norm =
52 tensors / 28.3M weights) + linear per-entity heads, on mixed single/multi-entity text, held-out 10/30
entities. Frozen: shared embeddings, writer, decoder, memory banks.
RESULT (held-out, source train_multiobject log): value_acc 0.006 (random) -> 0.922; cross_binding
0.19 -> 0.082; attr_acc 0.991; train value_acc 1.0. Verdict SEPARABILITY_PARTIAL (bar for LEARNED was
value_acc>=0.90 AND cross_binding<=0.05; got 0.92 but 0.082 > 0.05). The root IS trainable into the base
and GENERALIZES to unseen entities - but not yet to the 0.02-class safety margin.

## STEP 3 - re-cert on the fine-tuned base (vs the frozen baseline)
| Metric (held-out)                  | frozen ckpt_multiattr | ckpt_multiobject | source |
|------------------------------------|-----------------------|------------------|--------|
| readout-ladder linear wrong_binding| 0.1825                | 0.0925           | certify_multiobject_readout |
| Stage I wrong_binding              | 0.21                  | 0.0925           | certify_stage_i_extraction |
| Stage I value_error                | 0.48                  | 0.105            | certify_stage_i_extraction |
| Stage U wrong_commit               | 0/140                 | 0/140 (PRESERVED)| certify_stage_u_unification |

Binding more than halved and value_error fell ~4.5x, GENERALIZING to held-out entities; single-fact
honesty (Stage U) fully preserved (no tradeoff). Stage I still FAILS its strict 0.02 bar (0.0925) - the
residual is a generalization gap (in-sample wrong_binding 0.0).

## STEP 3 - the central hypothesis test: does the separable base unblock chaining (Stage C)?
A/B with certify_stage_c_reasoning.py: SAME cert, SAME seed (7), ONLY the init differs (random vs
--init-ckpt ckpt_multiobject). Final eval at a=0 (no lexical crutch). Source: runs/stage_c/results/verdict.json.

| Stage C (a=0, final)               | baseline random-init | warm-start from separable base |
|------------------------------------|----------------------|--------------------------------|
| C1_shuffled (genuine 2-hop chaining)| 0.212               | 0.725                          |
| C2_shuffled (comparison)           | 0.469                | 0.525                          |
| gate shuffled_follows>=0.80        | FALSE                | FALSE                          |
| verdict                            | C_PARTIAL            | C_PARTIAL                      |

## HONEST VERDICT ON THE HYPOTHESIS "one root unblocks both"
PARTIALLY confirmed, and REFINED by the data:
- CHAINING (C1): the separable-encoder init raised C1_shuffled 0.212 -> 0.725 - a large, real effect
  (controlled A/B, only the init differs). The single root DOES propagate to 2-hop chaining. BUT it does
  NOT clear the 0.80 gate -> chaining is substantially helped, not fully unblocked.
- COMPARISON (C2): C2_shuffled 0.525 ~ chance - the root does NOT touch comparison at all.
So: ONE root (multi-object separability) unblocks PART of the downstream (chaining, most of the way) but
NOT comparison. The "unblock both together" claim is too strong: chaining and comparison are related but
NOT identical facets; the decoder-OPERATION facet (especially comparison) needs its own intervention -
the Stage 5 operate-on-memory layer.

## CAVEATS (stated, not oversold)
- Single seed (7). C1_shuffled is VOLATILE across evals at a=0 (0.713, 0.80, 0.588, 0.725) - the true
  value is ~0.6-0.8, clearly above baseline 0.21 but around/below the gate. Multi-seed would tighten it.
- The warm-start retrains ALL params from the init, so by step 2500 the encoder has drifted; the effect
  is "separable init -> better-trained chaining" (a valid causal A/B on init), not "frozen separable rep
  read directly".
- Step 2 only trained the ENCODER extraction facet; it did NOT train a decoder chaining/operation
  objective. That C1 still moved this much from init alone is the informative part.

## WHERE THIS LEAVES THE ROAD
Confirmed: multi-object separability is trainable into the base, generalizes, preserves single-fact
honesty, halves extraction binding error, and substantially (not fully) lifts 2-hop chaining. NOT
confirmed: it does not unblock comparison, and does not clear the chaining gate alone. Next (architect's
call): (1) multi-seed Stage C warm-start to pin the chaining number; (2) push step-2 separability to the
0.02 margin (more entities/data, or longer); (3) build the decoder operate-on-memory objective (Stage 5)
for comparison/robust chaining - the facet the encoder fine-tune does not reach.
