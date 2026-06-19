# Implementation spec: Memory Tokenization + Canonical-Direct Shared Store (the first buildable piece)

Turns the PROVEN Stage U (internalize + honest mechanics on the model's own values, wrong_commit=0
in-distribution) into a memory you can actually LOAD A DOMAIN into: a learned codebook over the
internalized value vectors gives each value a discrete, scalable handle, and a canonical-direct path
lets the user write authoritative facts/rules straight into a shared, inspectable store. This does
NOT require the frontier (Stage C reasoning, Stage I book-scale extraction). It is the v1 substrate.

## The MemoryObject (the shared, inspectable unit)
```
MemoryObject:
  memory_token : int        # discrete handle from the learned codebook (same value -> same token)
  payload      : [768]      # the internalized content vector (w_value from Step B)
  entity       : str        # canonical entity id
  attribute    : str        # color | size | location | ... (or domain attribute)
  status       : committed | provisional | disputed | uncertain   # from the Stage U arbiter
  provenance   : user_canonical | model_internalized
  support      : {episodes:[...], source:str, trust:float}
```
Stored in an explicit, editable store (SQLite or a JSON-backed dict) the user can read/edit directly.
This explicitness IS the "shared" property: the user sees and edits what the model knows.

## Component 1 - Memory tokenizer (the learned codebook) [CRUX 1+2 resolved]
- A codebook C [K, 768], K starts 512 (growable). The token of a value vector v = argmin_k ||v - C_k||
  on the unit sphere (cosine). Trained with: (a) commitment loss ||v - sg(C_token)||^2 + ||sg(v) - C_token||^2
  (VQ-VAE style), (b) a SEPARATION term so distinct gold values map to distinct codes, (c) ANCHORED on
  the known gold value labels (we have them for the canonical vocab) so token<->value is a verified
  bijection on the trained vocab. So the token derives from the INTERNALIZED w_value (Step B), not from
  a text token - the line that keeps the axis-inversion (not regressing to copying).
- Decode: token -> C_token (a clean prototype) and/or the stored payload; value name via the model's
  aux head or the codebook<->value table.
- Capacity: K distinct tokens -> K distinct values; scales to thousands (vs the sealed organ's 37). This
  is the fix for the domain-campaign capacity wall.

## Component 2 - Canonical-direct path [CRUX 3 resolved]
- API: write_canonical(entity, attribute, value [, rule]) ->
  - encode "The {entity} is {value}." (or the location/size template) through the TRAINED model at
    lexical_alpha=0 to get the internalized payload w_value (so canonical and extracted objects share
    one representation space - reasoning later is provenance-agnostic).
  - tokenize the payload -> memory_token.
  - write a MemoryObject with status=committed, provenance=user_canonical, trust=1.0 (the user is
    authoritative; skip the multi-episode promotion gauntlet that provisional/extracted facts go through).
- RULES are objects too: a constraint/relation MemoryObject (e.g. (entity, "must_be", value) or a
  relational constraint), provenance=user_canonical, available to the reasoning layer (Stage C) later as
  operators. v1 stores them; C consumes them.

## Component 3 - Stage U mechanics on tokens [CRUX 4 - reuse the proven arbiter]
- The honest mechanics (committed/provisional/disputed + promote N=2 / retrograde M=2 / prune / reconcile)
  run with same/different = TOKEN EQUALITY (exact), not cosine. This is the discretization that made the
  unification clean. Provisional/extracted writes go through the gauntlet; canonical writes commit directly.

## Component 4 - The shared store (inspectable + editable)
- A store with: put(object), get(entity, attribute), list(), edit(entity, attribute, new_value),
  dump() -> human-readable. The user can audit and correct the model's memory directly. Edits re-tokenize.

## Falsifiable gates (pre-declared; this is a BUILD with verifiable correctness)
- G_TOKENIZER_BIJECTION: on the trained value vocab, same value (across entities/contexts) -> SAME token
  (0 splits) AND distinct values -> DISTINCT tokens (0 collisions). Report the collision/split counts.
- G_SCALE: with K=512 and a value vocab of N (e.g. 100-500), 0 collisions - demonstrating the capacity
  wall is gone (vs 37). Plot tokens-used vs N.
- G_CANONICAL_ROUNDTRIP: write_canonical(e,a,v) -> committed -> read back returns v exactly; provenance
  and status correct.
- G_ARBITER_ON_TOKENS: re-run the L1-L5 unification with token-equality identity -> wrong_commit=0 on
  the separable regime (re-confirm Stage U holds on the codebook tokens).
- G_INSPECTABLE: dump the store; edit one object; re-read reflects the edit; the arbiter sees the new token.
- G_NO_TEXT_LEAK: the token derives from w_value (internalized), NOT from the value's text token - verify
  by checking that two DIFFERENT surface forms of the same value (paraphrase) map to the same token, and
  that the token is unchanged if the entity/template changes (value-identity, not text-identity).

## Scope (honest)
- This is the SUBSTRATE -> loadable-domain-memory piece. It does NOT add reasoning (Stage C, the frontier)
  or book-scale raw extraction (Stage I, the frontier). With it, the user can load canonical domain facts
  + rules into an honest, inspectable, scalable memory the model grounds its answers in - the professional
  honest library. Reasoning over it (thinking IN memory) remains the separately-scoped frontier.
- Build order: tokenizer (verify G_TOKENIZER_BIJECTION + G_SCALE on the existing trained ckpt, CPU-light)
  -> store + canonical path -> re-confirm G_ARBITER_ON_TOKENS -> inspect/edit. Each gate verified before
  the next, no untested claims.
