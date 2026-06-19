# D_Cortex v1 - HONEST SCOPE (post Stage C, post shared-store)

## What v1 IS (verified, not aspirational)
v1 is the **SUBSTRATE track**: an exact, honest, auditable, self-revising **MEMORY STORE +
ADMINISTRATION** that the model grounds its answers in. Concretely, all proven:

- **Exact canonical storage.** When the user gives a value, v1 KNOWS it and assigns the memory token
  DIRECTLY from the value (SharedMemoryStore.write_canonical). No internalization drift. The codebook
  bijection floor (~92%, internalization-limited) applies ONLY to the not-yet-built EXTRACTION path
  (Stage I), never to canonical writes.
- **Honest mechanics on the model's own memory.** committed / provisional / disputed with
  promote / retrograde / prune / reconcile. Stage U proved wrong_commit = 0 on the whole L1-L5 regime.
- **Arbiter-on-tokens, exact on canonical.** Running the Stage U arbiter with canonical token identity
  gives wrong_commit = 0 (140/140; certify_shared_store.py). No internalization asterisk on canonical.
- **Scalable.** The memory tokenizer removed the sealed organ's 37-token wall (G_SCALE: 200 distinct
  values -> 200 tokens, 0 collisions). v1 can hold a real domain, not a toy vocabulary.
- **Inspectable + editable.** The store is shared: the user reads it, edits it, and the edit is
  reflected exactly (G_INSPECTABLE). The user sees and corrects what the model knows.
- **No out-of-memory hallucination.** Answers are grounded in committed memory; unwritten -> abstain.

This is strictly MORE than RAG: RAG retrieves text chunks; v1 holds value-identity memory objects with
provenance, trust, status, and honest self-revision the user can audit and correct.

## What v1 is NOT (the claim Stage C undermined - do NOT carry it)
v1 is **NOT "the model reasons over your domain"**. Stage C MEASURED this and REFUTED it:
- C1 (2-hop relational chaining): the high single-option accuracy was a **shortcut**, refuted by the
  pre-declared shuffled control (multi-distractor at chance). The model does not chain through memory.
- C2 (comparison): at chance.
- The thinking MECHANISM (operate-on-memory: a working scratchpad, multi-hop chaining) is UNBUILT on
  the single-pass decoder. It is NOT proven untrainable - it is **Stage 5, the separate frontier.**

Do not describe v1 as reasoning, inference-over-memory, or "thinks in memory". Those are Stage 5.

## Two tracks, kept separate
- **SUBSTRATE track (v1, this work):** scalable, value-based, honest STORAGE + ADMINISTRATION. DONE
  and verified. The tokenizer makes the store good; it does not touch the thinking gap.
- **THINKING track (Stage 5, untouched):** the operate-on-memory layer. The frontier C localized.

## Deferred (do not start early)
- **VQ codebook refinement (b):** would lift the EXTRACTION-path bijection from ~92% toward >=0.99.
  Pointless until extraction exists. Build it WHEN attacking Stage I (extract facts from raw text),
  not before. Canonical writes never needed it.
- **Stage I (extraction):** turn raw input into memory objects automatically. The ~8% floor lives
  here; VQ is its tool.

## One-line framing for any report
"v1 = an exact, honest, auditable, self-revising memory store + administration the model grounds
answers in (more than RAG); reasoning over that memory is Stage 5, unbuilt and not claimed."
