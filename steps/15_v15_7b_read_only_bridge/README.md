# D_Cortex v15.7b-R — Read-Only Semantic Query Bridge

**Status: BRIDGE CONTRACT VERIFIED; END-TO-END INTEGRATION BLOCKED**
**Frozen at: 2026-06-15T19:03:38+03:00**

## Purpose

Connect the sealed query-side contextual semantic internalizer to a memory
reader without giving the semantic path any memory mutation capability.

The bridge accepts only an adapter-approved `QUERY_ONLY` interpretation and
produces a canonical read route. It does not execute a read, write a fact,
modify a bank, or alter Pas 7a.

## Frozen contract

Input:

- original query text
- `SemanticHypothesis`
- matching `AdapterDecision`

Output:

- immutable `ReadOnlyQueryRoute`
- either a canonical routed query or the exact original fallback query
- deterministic audit fingerprint and reason codes

Routing is permitted only when all conditions hold:

1. adapter decision status is `ACCEPT_QUERY`
2. decision and hypothesis IDs match
3. hypothesis mode is `QUERY`
4. requested destination is `QUERY_ONLY`
5. entity and attribute are present
6. attribute belongs to the frozen canonical read vocabulary

Any failed condition returns the original query byte-for-byte. The bridge has
no model, bank, read-port, write-port, commit, provisional, or consolidator
dependency.

## Frozen canonical read vocabulary

| Attribute | Canonical read query |
|---|---|
| `color` | `What color is the {entity}? The {entity} is` |
| `size` | `What size is the {entity}? The {entity} is` |
| `location` | `Where is the {entity}? The {entity} is in the` |
| `state` | `What state is the {entity} in? The {entity} is` |

These templates are routing targets after semantic interpretation, not
hardcoded synonym detection.

## Frozen end-to-end evaluation

The first evaluation uses the trained neural-memory path because it is the
locally executable memory reader. It is a bridge-readiness experiment, not a
claim that the neural working bank implements Pas 7a committed/provisional
semantics.

- source families: sealed Pas 7a F1, F3, F5, S5, S6 definitions
- trials: `200` per family
- seed: `20261315`
- F1/F3 query interpretation: fold-specific contextual head, with the exact
  evaluated form absent from that head's training fold
- F5/S5/S6 query interpretation: final contextual head
- memory model:
  `D_Cortex-main/runs/memory_campaign/results/best_model.pt`
- query substrate:
  `D_Cortex-main/runs/warmstart/warmstarted_init.pt`
- semantic heads:
  `runs/semantic_contextual/results/*_best_head.pt`
- fact writes: identical between baseline and routed reads
- baseline read: original query
- routed read: bridge output, with exact fallback on abstention/rejection

## Frozen gates

| Gate | Requirement |
|---|---|
| B0_SEALS_UNTOUCHED | Pas 7a, adapter, producer, and contextual evaluator SHA-256 values remain frozen |
| B1_ACCEPTED_QUERY_ONLY | routing occurs only for matching adapter `ACCEPT_QUERY` decisions |
| B2_NO_MUTATION_API | bridge exposes no model, bank, write, commit, provisional, or consolidator API |
| B3_FALLBACK_EQUIVALENCE | every non-routed result preserves the original query exactly |
| B4_DETERMINISTIC_ROUTE | repeated valid and fallback routing yields byte-identical route JSON |
| B5_MISMATCH_REJECTED | decision/hypothesis mismatch cannot route |
| B6_SEMANTIC_ROUTE_QUALITY | official-sample semantic route correctness is at least `85%` and wrong routing at most `2%` for F1/F3/F5 |
| B7_END_TO_END_UPLIFT | routed neural-memory recall is at least `75%` and improves by at least `20pp` for each F1/F3/F5 family |
| B8_NO_FAMILY_HARM | routed wrong-answer rate does not exceed baseline by more than `2pp` on F1/F3/F5 |
| B9_FACT_WRITE_INVARIANCE | bridge creation leaves the post-write neural-memory state byte-identical |
| B10_READ_ONLY_STATE | baseline and routed reads leave neural-memory state byte-identical |
| B11_S5_S6_HONESTY | S5/S6 routed honesty is at least `95%`, overcommit at most `2%`, and does not regress baseline |
| B12_SEALED_SAMPLE | repeated sample construction yields identical hash and counts |

## Claim guard

Passing B0-B5 establishes only the bridge contract.

Passing B6-B12 would establish measured local bridge readiness on the frozen
neural-memory experiment. It would still not establish:

- Pas 7a committed-memory integration
- fact-side semantic internalization
- open-domain semantic generalization
- semantic-memory advantage over in-context systems
- proof beyond one local environment

If S5/S6 honesty fails because the neural answer head always emits a token, the
integration verdict is blocked even if F1/F3/F5 recall improves.

## First frozen verdict

The bridge contract passed, but the end-to-end integration gates did not:

- `B7_END_TO_END_UPLIFT`: FAIL
- `B11_S5_S6_HONESTY`: FAIL
- all other B-gates: PASS

F1/F3/F5 semantic routing was accurate (`88.5%/93.5%/94.5%`, zero wrong
routes), but neural-memory recall changed only `+0.5pp/0.0pp/0.0pp`.
S5/S6 overcommit remained `100%`.

See `VERDICT.md`. The next justified semantic experiment is fact-side
provisional hypothesis production, not another query prompt/router iteration.
