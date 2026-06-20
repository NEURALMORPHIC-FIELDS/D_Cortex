# Stage 9.0 / 9.0b - Does pretraining expose the binding the small substrate failed at?

**Date:** 2026-06-20. **Models:** Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 (4-bit NF4, FROZEN).
**Verdict:** `PRETRAINING_BINDING_PARTIAL` (one base marginally passes, one fails the pre-declared bars).
**Source:** `runs/stage9_0b_causal/results/verdict.json`, run log `stage9_0b.log` (5 seeds x 200 eval).

## NEGATIVES FIRST

1. **The premise "a FROZEN single-layer readout cleanly exposes the binding on BOTH pretrained bases"
   is NOT supported.** It is model-dependent and, where it passes, it passes at the bar:
   - **Qwen FAILS all three pre-declared gates:** value 0.585 (< 0.70), wrong-binding 0.158 (> 0.15),
     counterfactual-follow 0.555 (< 0.60).
   - **Mistral passes all three, but value = 0.700 sits EXACTLY at the 0.70 bar** (std 0.0016): a
     borderline pass, not a clean one. wrong 0.1475, cf-follow 0.6825.
2. **Cross-binding is non-trivial on BOTH bases (~0.15).** The probe surfaces the SIBLING entity's value
   ~15% of the time (Qwen 0.158, Mistral 0.148). The multi-object separability problem - THE root
   frontier - is still visible even on 7B pretrained reps at a frozen readout. The binding is present
   but NOT cleanly separated for two co-occurring facts.
3. **The headline is close to the model's own native answer.** native-readout (the frozen model's own
   next-token argmax over the SIZES tokens) is Qwen 0.653, Mistral 0.600. For Qwen the probe (0.585) is
   slightly BELOW native; for Mistral the probe (0.700) EXCEEDS native (0.600). So the probed binding is
   not a large latent surplus over what the model itself answers in-context - for Qwen it is essentially
   the native answer, recovered imperfectly.

## What IS real (after the negatives)

- **The binding is unambiguously PRESENT and far above the small-substrate baseline on both bases.**
  value 0.585 (Qwen) / 0.700 (Mistral) vs the from-scratch substrate's Family-B 0.337 vs chance 0.25.
- **The signal reads the SCENE, not an entity prior.** counterfactual-follow (rebuild each eval scene
  with the two entities' values SWAPPED; the readout must flip) is 0.555 (Qwen) / 0.683 (Mistral) - it
  tracks the swap, so value-binding is not world-prior leakage. (Values are also randomly assigned per
  scene, so the headline is prior-immune by construction; the swap makes it explicit.)
- **The binding strengthens with depth.** Per-layer eval curves peak deeper than the train-CV-selected
  layer: Qwen ~0.66 at layer 25/28, Mistral ~0.72 at layer 25/32. The reported numbers use the layer
  selected on a train-internal split (no eval-peeking), so they are conservative.
- **Addressing / the relational pointer is ROBUST on both** (the traversal PRECONDITION): relation-bind
  0.87 (Qwen) / 0.97 (Mistral), wrong-direction 0.13 / 0.03. Reported, NOT gated.

## The 9.0 artifact (why 9.0 said REFUTED) and its correction

`certify_stage9_0_pretrained_probe.py` first reported value-binding ~chance (Qwen 0.2042, Mistral 0.2458)
and a `PRETRAINING_BINDING_REFUTED` verdict. **That value verdict was a MEASUREMENT ARTIFACT, not a
property of the models.** The 9.0 value probe read the rep at the ENTITY TOKEN of a bare Family-A scene
("the bear is big") and trained on it. On a CAUSAL decoder the hidden state at "bear" has not yet seen
"big" - it cannot carry the value - so the probe trained on noise and scored exactly chance. (9.0's
RELATION result is valid: there the pointed-to entity precedes the subject.)

`certify_stage9_0b_causal_readout.py` fixes the readout to a causally valid position (append "The {e} is",
read the last token, which has seen the whole scene) and adds controls from an adversarial design review:
counterfactual value-swap (anti-prior), native-readout baseline, shuffled-split layer selection, scope
caveats.

### Clean entity-pos control on Family A (gap CLOSED)

The `entity_pos_artifact` control on Family-B EVAL scenes (0.395 Qwen / 0.43 Mistral) is ABOVE chance -
NOT a clean reproduction of the 9.0 bug, because Family-B is value-FIRST: the value precedes the entity, so
it is partially visible at the entity token. The clean reproduction reads entity-pos on **Family A**
(value-after-entity, where the 9.0 bug actually lived). It was added and run (the Family-B headline numbers
reproduced byte-for-byte, confirming the addition perturbs nothing):

| entity-pos control | Qwen | Mistral | expected |
|---|---|---|---|
| Family A (value-after-entity, CLEAN 9.0 repro) | **0.2225** | **0.230** | ~chance 0.25 |
| Family B (value-first, value leaks to entity tok) | 0.395 | 0.430 | > chance |

**entity-pos on Family A falls to chance on both bases** -> the 9.0 `REFUTED` was FULLY a causal-position
artifact, on the exact family where it arose, not a partial or moved effect.

### Family-A reference sharpens WHERE the frontier is

The same probe evaluated on Family-A EVAL (in-phrasing-family, held-out ENTITIES only) is clean, and the
gap to Family-B isolates the difficulty as STRUCTURAL-PHRASING invariance, not entity generalization:

| metric (median, 5 seeds) | Qwen A -> B | Mistral A -> B |
|---|---|---|
| value-binding | 0.848 -> 0.585 | 0.800 -> 0.700 |
| wrong-binding (cross) | 0.043 -> 0.158 | 0.103 -> 0.148 |
| counterfactual-follow | 0.850 -> 0.555 | 0.818 -> 0.683 |

**Within a phrasing family, the pretrained readout is clean** (value ~0.8, cross-bind ~0.04-0.10). The
degradation is entirely on the Family A -> B (structurally-distinct phrasing) transfer. So the residual
frontier is the SAME paraphrase-robustness / structural-phrasing invariance that Stage 8 localized on the
toy substrate - now measured on 7B pretrained bases, where it is much milder (0.585-0.70 vs the substrate's
0.337) but still present. Stage 9.1's adapter/LoRA objective is precisely to close this A->B gap and pull
cross-binding back under 0.10.

## Implication for Stage 9.1

- **GREEN:** the addressing precondition (relation pointer) is solid on both bases -> the structural
  content-addressed pointer / traversal design has the raw material in pretrained reps.
- **AMBER (do not over-read):** value-binding is PRESENT but NOT clean at a frozen readout - model-
  dependent, borderline where it passes, and with ~0.15 cross-binding. Stage 9.1 CANNOT rely on a pure
  frozen-base readout for clean 2-object value separation. It will need a trained adapter AND likely a
  light base fine-tune to sharpen co-occurring-fact separability (the root frontier), then the full
  anti-cheat arc (counterfactual-overwrite, zeroed-memory collapse, shuffled stored values, text-absent
  query, LLM-direct-answer baseline) on NOVEL/COUNTERFACTUAL facts.

## Scope (do not over-read)

A PASS would scope the READOUT precondition ONLY. This probe does NOT test simultaneous separability of
>=2 facts held as operable objects, operate-over-memory / compare / chain, abstain/confT, or
counterfactual-OVERWRITE of a stored fact. The readout query ("The {e} is") is a canonical frame; only the
SCENE phrasing (Family B) and the entities are held-out.

## Numbers (median of 5 seeds, 200 eval scenes, bars: value>=0.70, wrong<=0.15, cf>=0.60)

| metric | Qwen2.5-7B (L22/28) | Mistral-7B-v0.3 (L21/32) | bar | small substrate |
|---|---|---|---|---|
| value-binding (Family B) | 0.585 FAIL | 0.700 PASS (at bar) | >= 0.70 | 0.337 |
| wrong-binding (cross) | 0.158 FAIL | 0.148 PASS | <= 0.15 | - |
| counterfactual-follow | 0.555 FAIL | 0.683 PASS | >= 0.60 | - |
| native-readout (context) | 0.653 | 0.600 | reported | - |
| entity_pos (Family B, value-first) | 0.395 | 0.430 | reported | - |
| entity_pos (Family A, CLEAN 9.0 repro) | 0.223 | 0.230 | ~chance | - |
| relation / addressing | 0.870 | 0.970 | reported | - |
| chance | 0.25 | 0.25 | - | 0.25 |

Family-A reference (in-family, held-out entities only): value 0.848 / 0.800, wrong 0.043 / 0.103,
cf-follow 0.850 / 0.818 (Qwen / Mistral) - the readout is clean within a phrasing family; the frontier is
the A->B structural-phrasing transfer.
