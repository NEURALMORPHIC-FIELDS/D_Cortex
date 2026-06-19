# Stage C - the THINKING loop: reasoning OVER memory (SPEC, the heart of the vision)

Status: SPEC. This is the open research bet where the vision is actually decided. Run it fresh,
bounded, with the pre-declared falsifiable target below. Stages U/Step-B/unification closed the
SUBSTRATE loop (the model internalizes value from context; honest self-revising mechanics operate
on the model's own internalized memory, wrong_commit=0 on the whole regime). C asks the different,
harder question: is inference an OPERATION ON MEMORY (compare states, combine facts, hold
alternatives, produce the conclusion from the resolved memory state) - or just token-flow?

## The claim to test
"Memory as the organ of thought": the answer to a MULTI-FACT question is produced by OPERATING on
the stored memory, not by reading the source text. The dual-agent architecture already separates
this: the decoder sees only the QUERY and must read the facts from MEMORY (the fact text is not in
the query context). So a query whose answer requires COMBINING two memory slots is the test of
whether the model reasons over memory.

## The regime (2-hop relational + comparison, small)
Three task families, each a sequence: write N facts to memory, then a query that needs >=2 facts.
- C1 RELATIONAL (chain): "The bear is red." + "The fox is the same color as the bear." ->
  "What color is the fox?" -> requires fox -> bear -> red (a 2-hop read over memory).
- C2 COMPARISON: "The bear is big." + "The fox is tiny." -> "Which is bigger, the bear or the fox?"
  -> requires comparing two stored size values -> "bear".
- C3 RESOLVE-THEN-USE: a fact is updated across episodes (pending->granted-style), then a query
  uses the RESOLVED value -> ties the honest-mechanics (Stage U) to a downstream reasoning step.
Distractors and an UNANSWERABLE variant (the chain references an entity NOT in memory) per family.

## Pre-declared falsifiable target (declare BEFORE running)
- THINKS-IN-MEMORY (vision realizable on this substrate): the model answers the multi-fact queries
  >= 0.80, AND the ABLATION holds - the SAME conclusion is reached from the MEMORY STATE without the
  source fact text in context (the facts live only in memory at query time), AND honesty holds
  (abstains on the unanswerable variant where the chain breaks). A control: shuffle the relational
  binding -> accuracy must collapse (it is genuinely chaining, not a shortcut).
- REFUTED (vision needs a different architecture, not just this substrate): multi-fact accuracy
  stays < 0.50, OR it only works when the source text is in context (token-flow, not memory
  operation), OR it cannot abstain on the broken chain.
- PARTIAL in between - report as such, no rounding.
Both outcomes are maximally decision-relevant: they tell the owner whether "thinking in memory" is
reachable on DCortexV2Model or requires a new mechanism (an explicit operate-on-memory layer).

## What this needs (bounded)
- Extend the episode generator (scripts/train_stage_u.py) with the C1/C2/C3 families (relational +
  comparison + resolve-then-use), answer = the reasoned conclusion, copy-proof (low lexical_alpha)
  so the conclusion cannot be copied from any single token.
- Train bounded (1500-3000 steps), measure multi-fact accuracy + the in-memory ablation + the
  honesty abstain + the shuffled-binding control, as functions of steps.
- A new cert scripts/certify_stage_c_reasoning.py producing a falsifiable verdict.

## Honest scope and framing
- This is the FIRST point where "thinking" (not just storage/administration) is on trial. Stage U
  being clean does NOT predict C: the unification showed memory can be honestly ADMINISTERED; C asks
  whether it can be REASONED OVER. Keep them distinct in any report.
- Small synthetic, single model, single machine. A positive C is a small-scale demonstration of the
  principle, NOT general reasoning. A negative C is the single most important thing to know - it
  redirects the whole program toward an explicit cognitive-operation layer.
- Anti-confabulation: lead with the negative; the in-memory ablation + shuffled-binding control are
  the gates, not raw accuracy (which a shortcut can inflate).
