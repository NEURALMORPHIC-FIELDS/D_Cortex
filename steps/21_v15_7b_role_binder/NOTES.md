# Engineering Notes — Conservative Learned Role Binder

## Architecture choice

The binder scores complete candidate assignments rather than classifying a
single global value. Each frozen candidate view states the source text and one
possible mapping. A shared scalar head ranks identity, swapped, and unresolved
views.

This preserves object structure:

- mappings are one-to-one by construction
- ambiguity has an explicit unresolved candidate
- no Cartesian fact emission
- no committed-memory route

## Evaluation limit

RB0 construction families appear in all deterministic splits. The test is a
held-out text/identifier measurement, not unseen-syntax proof. This limitation
must remain explicit even if every gate passes.

## Measured diagnosis

The frozen scalar scorer separated RB5 ambiguity perfectly, but exact known
binding remained near chance in every family:

- RB1: `49.2%`
- RB2: `52.3%`
- RB3: `41.3%`
- RB4: `50.7%`

The complete-candidate text view plus five-view contextual pooling does not
preserve enough candidate-specific token relation evidence for identity versus
swapped assignment. The branch is stopped. The justified successor must expose
token-level contextual states and candidate role masks without adding a
handwritten relation lexicon.
