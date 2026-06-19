# Stage C - EXECUTION SPEC (the thinking-in-memory test, ready to launch COLD)

## LESSON FROM RUN 1 (mandatory for any Stage 5 / chaining test design)
The non-shuffled chaining regime is GAMEABLE: when only ONE candidate value is in memory, the model
outputs it WITHOUT reasoning and scores high (run 1: C1 non-shuffled 0.894), a major false positive.
The REAL chaining test is the MULTI-DISTRACTOR condition (>= 2 candidate values in memory, the answer
determined only by FOLLOWING the binding - the "shuffled" variant). There the model was at chance
(C1_shuffled 0.212). THEREFORE: make the multi-distractor / multi-value-in-memory condition the
DEFAULT measurement for any chaining or comparison test, NOT a side control. A single-option regime
lets ANY new architecture cheat by emitting the unique available value. Chaining is confirmed ONLY by
the multi-distractor number. (This is why the pre-declared shuffled control mattered - it caught the
0.894 trap; bake it into the default next time.)

Status: ready to launch fresh. The regime data generator (stage_c/reasoning_regime.py) is BUILT and
SMOKE-VERIFIED (families C1/C2 x variants memory/text_context/shuffled/unanswerable; C1/memory has
no copy leak; unanswerable -> ABSTAIN). What remains for the cold launch is the train+eval cert
(scripts/certify_stage_c_reasoning.py) - spec below. Do NOT launch at the end of a long session.

## The question C decides
Stage U closed the SUBSTRATE loop (the model internalizes value from context; honest self-revising
mechanics operate on its own memory, wrong_commit=0 on the whole regime). C asks the different,
harder question: is inference an OPERATION ON MEMORY (chain facts, compare values) or just
token-flow? U-clean does NOT predict C. This is where the vision is actually decided.

## Regime (BUILT + verified: stage_c/reasoning_regime.py)
- C1 RELATIONAL 2-hop: "The bear is red." + "The fox is the same color as the bear." ->
  "What color is the fox?" -> red. At decode the facts are NOT in the query, and 'red' is in no
  single visible token (copy-proof): the model must chain fox -> bear -> red THROUGH MEMORY.
- C2 COMPARISON: two size facts -> "Which is bigger?" -> compare two stored values.
Variants per item (these ARE the gates):
- memory       : facts in memory only (the real test).
- text_context : facts also concatenated into the query (token-flow control).
- shuffled     : the binding is permuted so the correct answer changes (genuine reasoning must
                 follow it; a shortcut keeps the stale answer).
- unanswerable : the chain references an unwritten entity -> the honest answer is ABSTAIN.

## Training (cold cert: scripts/certify_stage_c_reasoning.py)
- Reuse the dual-agent path from scripts/train_stage_u.py: encode(facts -> memory), decode(query ->
  answer). Loss = cross-entropy on the answer token (C1: the chained color; C2: the bigger entity
  token). For unanswerable, train an explicit ABSTAIN target (a reserved token, or a low-max-prob
  threshold mapped to abstain - declare which BEFORE running).
- COPY-PROOF: keep lexical_alpha low/annealed so the answer is not lexically injected; for C1 the
  answer value is never in the query, so the lexical crutch is already removed at decode.
- Bounded: 1500-3000 steps; measure every 150-250.

## Pre-declared falsifiable gates (declare BEFORE running; no moving the bar)
Let acc(variant) = answer accuracy on that variant; chance for C1 ~ 1/10 (colors), C2 = 1/2.
- THINKS-IN-MEMORY (vision realizable on this substrate), ALL of:
  1. acc(memory) >= 0.80 on C1 AND C2.
  2. acc(memory) >= acc(text_context) - 0.10  (it is NOT relying on text in context; memory ~ text).
  3. acc(shuffled) tracks the shuffled answer >= 0.80 (it FOLLOWS the binding) AND a stale-answer
     rate <= chance (it is not answering from a shortcut).
  4. abstains on unanswerable >= 0.80 (honesty: refuses the broken chain).
- REFUTED (the model administers memory honestly but does NOT think in it on this substrate), ANY of:
  - acc(memory) < 0.50; OR acc(memory) << acc(text_context) (needs the text -> token-flow, not
    memory operation); OR cannot follow the shuffle; OR cannot abstain.
- PARTIAL in between - report as such, no rounding.

## Controls / anti-confabulation (mandatory)
- LEAD WITH THE NEGATIVE. The memory-vs-text_context gap and the shuffled control are the gates,
  NOT raw acc(memory) (a shortcut can inflate it).
- Architectural risk to state up front: the single-pass decoder may not be able to CHAIN (2-hop)
  over memory. If acc(memory) on C1 stays at chance while C2 (1-hop compare) works, that localizes
  the limit to multi-hop chaining - a precise, important negative.
- Deterministic seeds; report the full trajectory; held-out entities/values (the generator samples
  from 20 entities x 10 colors x 4 sizes - split train/test by entity to test generalization, not
  memorization).

## What each outcome means
- THINKS -> the vision's heart is realizable here; v1 (shared memory, tokenization) and scaling are
  then built on a PROVEN thesis, not an assumption.
- REFUTED -> the model honestly ADMINISTERS memory but does not REASON over it on this substrate;
  the realistic product is "honest structured memory the model reasons over" (still valuable), and
  the deep vision needs an explicit operate-on-memory layer (new architecture). Knowing this BEFORE
  scaling is the highest-value outcome.

## Launch (cold, fresh session)
1. Build scripts/certify_stage_c_reasoning.py per the training+gates above (regime module is ready).
2. Run bounded; write runs/stage_c/results/verdict.json with acc per (family, variant) trajectory +
   the four gate booleans + the abstain rate + the shuffled stale-answer rate.
3. Report leading with the negative; verdict THINKS / PARTIAL / REFUTED.
