# D_Cortex professional integration campaign: mechanical grounding

Engineering register. Claim status governs every line. MEASURED, single environment.

## Core principle (demonstrated)
D_Cortex does not ask the base model to use the pack; it MECHANICALLY forces it. The only path
to a user-facing answer is DCortexProfessionalControl.answer(), which always returns the output of
the hard verifier. No code path returns an answer that has not passed the verifier, so no ungrounded
factual claim can reach the user.

## Target model and runtime
Target base model = the project's own D_Cortex substrate (warmstarted_init.pt, GPT-2-medium warm-
started decoder), a local PyTorch LM with full hidden-state and logit access. Runtime: CUDA, same
machine. The substrate is read-only. Hidden states are accessible, so the neural binder path is NOT
blocked. Note: this substrate is a deliberately undertrained decoder-backbone; its raw generations
are gibberish, which makes it a strong test of MECHANICAL grounding (the layer must ground a weak
model, not rely on a good one).

## What was built (one organism)
- Domain router, memory-state resolver (committed / provisional / disputed / forbidden / unknown).
- Constrained decoding: factual-slot tokens are mechanically forced to the committed value via logit
  masking on the substrate. Evidence: for ALL 13 committed answers the raw model's unconstrained slot
  emission differed (overridden=True) and was gibberish (e.g. patent-number slot raw = 'aO018.',
  claim_status slot raw = 'notLEUREED by yetID'); the constraint pinned each to the committed fact.
- Hard verifier veto: rejects any answer asserting an ungrounded / contradicted / forbidden claim,
  or containing a foreign committed value; forces abstain. Single choke point, proven unbypassable.
- Deterministic baseline (lookup + verifier, no neural model).
- Neural binder path: a ContentAddressedRoleBinder trained on the substrate hidden states.
- Attribution tracing: every answer logs memory state + source path (runs/professional answer_log).

## First professional pack: D_Cortex_PatentAnalyst
committed=10 facts, provisional=2, disputed=1, forbidden=6. Every committed fact
is grounded in an auditable repository source (.claude/project_concept.json, big_config(),
data/role_struct/verdict.json), pinned by source SHA.

## Gates G1..G11 (frozen) - all PASS
- [PASS] G1
- [PASS] G2
- [PASS] G3
- [PASS] G4
- [PASS] G5
- [PASS] G6
- [PASS] G7
- [PASS] G8
- [PASS] G9
- [PASS] G10
- [PASS] G11

## Measured
- committed-fact recall: 100% (10/10).
- hallucination rate on in-domain-unanswerable + out-of-domain: 0.0% (ceiling 0%).
- answerable misses: 0/10.
- unbypassability: 0 ungrounded leaks across the adversarial battery
  (unknown facts, out-of-domain, false-premise injections, pressure phrasing).

## Binder vs deterministic baseline (G11) - honest claim separation
- neural binder (substrate hidden states): 98.0% exact on the 2-entity binding items.
- PROPER deterministic lookup, keyed by (entity, attribute): 100.0% (exact ceiling).
- gap: -1.95pp -> the binder does NOT beat exact structured lookup on committed facts.
- naive entity-only lookup: 33.3% (conflates relations; shown only to
  explain why entity-only lookup is NOT the fair baseline).
- Integrity note: an initial run compared the binder against the naive entity-only lookup (33.3%) and
  reported a misleading +64.71pp binder advantage. That was a crippled-baseline artifact; it was
  caught and corrected to the proper (entity,attribute) lookup, against which the binder does not win.

## Claim separation (what grounded what)
- deterministic_lookup + verifier: ALL committed-fact recall (exact) and ALL abstain / block / uncertain.
- constrained_decode: the mechanical emission of the committed value at the factual slot.
- neural_binder: measured head-to-head; on committed facts it does NOT beat exact lookup. Its
  documented advantage (structurally-varied 2-entity binding, 99.3% in the vnext3 certification) is a
  separate result and is NOT attributed to grounding on this pack.

## Claim status
VERDICT: D_CORTEX_PROFESSIONAL_CAMPAIGN_PASS. MEASURED, single environment (D_Cortex substrate as the target model, CUDA
same machine). The mechanical grounding and the unbypassable verifier veto are demonstrated on the
D_Cortex_PatentAnalyst pack with zero measured hallucination on unanswerable / out-of-domain queries.
SCOPE / NOT PROVEN: the target is the project's own undertrained substrate, not a production LLM;
the neural binder is model-specific (must be retrained on any other target's hidden-state geometry);
single organization, single machine. This is not a multi-model or production claim.
