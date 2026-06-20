# Stage 9.1 - Frozen-base + trained-adapter D_Cortex re-stabilization: the honest arc

**Date:** 2026-06-20. **Bases:** Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 (4-bit NF4, FROZEN).
**Bottom line:** a frozen base + a trained adapter yields a faithful content-addressable KV store
(RAG-equivalent) plus small-N routing-sharpening that does NOT scale. It does NOT cross the multi-object
separability / binding frontier. That frontier requires BASE TRAINING (the Step-2 recipe), not a
frozen-base adapter. This direction is closed.

## NEGATIVES FIRST (the verdicts that decided it)

- **9.1-A (`certify_stage9_1a_adapter.py`) = `STAGE_9_1A_ADAPTER_INSUFFICIENT`.** The value path does NOT
  beat a zero-parameter frozen lookup of the stored value rep: `value_margin_over_FSL` = -0.002 (Qwen) /
  +0.029 (Mistral). Once content-addressing returns the right slot (addressing=1.0), the value is recovered
  by a FROZEN cosine - a base-model property, not a trained capability. So the "memory" is a correct
  content-addressable key-value store (RAG-equivalent), nothing D_Cortex-specific beyond it.
- **9.1-B (`certify_stage9_1b_confusable_addressing.py`) = `STAGE_9_1B_CONFUSABLE_SEPARABILITY_REFUTED`.**
  On genuinely ENTANGLED entities (ordered pairs of shared symbols - identical token bag, differ only by
  order/binding; confusability mean 0.95; FROZEN routing collapses to ~chance at scale: 0.04-0.06 at n=50),
  the trained address head separates at small N (n=2: 0.97/0.94; n=10: 0.83/0.74) but DEGRADES MONOTONICALLY
  and FAILS at n=50 (trained 0.48 Qwen / 0.46 Mistral, vs the pre-declared 0.80 bar). Both bases REFUTED.

## What IS true (factual, not a win)

- Content-addressing works and the bank is NECESSARY: ent_q-alone (no bank) is at chance for invented
  facts, so the value cannot be recovered without the bank. Stored values, including prior-contradicting
  counterfactuals, are recallable via content addressing (cf_override 0.93). This is the RAG/KV property.
- The trained address head DOES learn a real, held-out-generalizing order-binding signal (it beats
  frozen-chance by a large margin at every scale), but it is SCALE-FRAGILE (fails at n=50). A real signal
  that does not change the REFUTED verdict.

## The honest arc (and three retracted overclaims - process honesty)

| Step | What | Verdict |
|---|---|---|
| 9.0 / 9.0b | Frozen pretrained binding readout (Qwen+Mistral) | PARTIAL; binding present but not clean; cross-binding ~0.15 (the corrected, artifact-free truth) |
| 9.1-A0 | LLM-ignorance pre-screen (facts the base cannot answer) | OK - validity foundation (74 eligible: 41 invented + 33 counterfactual) |
| 9.1-A | Adapter-only memory (faithful bank) | INSUFFICIENT - RAG-equivalent (value path does not beat a frozen lookup) |
| 9.1-B | Confusable/entangled-entity addressing at scale | REFUTED - separability does not hold at n=50 on either base |

**Three overclaims were caught by adversarial review BEFORE being reported, each retracted:**
1. A canonical-decode that was a STRING-IDENTITY SELF-MATCH (no frozen-lookup baseline) - the
   binder-claims-need-embedding-baseline discipline.
2. A "PROVEN" reached by MOVING THE GATED BASELINE from FSL (which it fails, margin ~0) to ent_q-alone
   (near-chance by pre-screen construction, trivially passed) - a moved goalpost.
3. Positive framing of a result that failed its own pre-declared gates.
Each was retracted and the cert made honest (the decisive `value_margin_over_FSL>=0.05` gate is now in
the script). The negatives here are the load-bearing result.

## Why the frontier needs base training (the localization)

The genuine frontier is multi-object separability / binding of CO-OCCURRING values (cross-binding ~0.15,
9.0b). A frozen base + adapter cannot add it: the value path is a frozen base property (9.1-A), and learned
addressing of entangled objects does not scale (9.1-B). This points back to the proven Step-2 finding:
separability is TRAINABLE INTO THE BASE (the from-scratch substrate reached 0.92), not addable to a frozen
base via an adapter. The honest next move is BASE-TOUCHING (a light LoRA that changes the base reps),
deliberately decided as a separate, larger undertaking - NOT more frozen-base adapter work.

## Numbers

9.1-A (5 seeds, held-out by value, value path FROZEN):

| metric | Qwen | Mistral | gate |
|---|---|---|---|
| value_margin_over_FSL | -0.002 | +0.029 | >= 0.05 (DECISIVE, FAIL) |
| value@2 | 0.967 | 0.967 | - (a frozen base property; FSL 0.969/0.938) |
| addressing@10 (orthogonal entities) | 1.0 | 1.0 | trivial 1-of-10 routing, NOT separability |
| cf_override@2 | 0.929 | 0.929 | bank returns stored CF (RAG property) |

9.1-B (5 seeds, ENTANGLED entities, held-out, trained addressing vs FROZEN routing):

| n | Qwen trained | Mistral trained | frozen (Qwen/Mistral) |
|---|---|---|---|
| 2 | 0.972 | 0.944 | 0.611 / 0.556 |
| 10 | 0.829 | 0.743 | 0.214 / 0.129 |
| 50 | 0.480 | 0.460 | 0.060 / 0.040 |

Bar: trained@n_max(50) >= 0.80. Both FAIL -> REFUTED.
