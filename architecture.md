# D_Cortex v2.0-alpha: Architecture Documentation

> **Sealed milestone**: v15.7a (Pas 7a, 2026-04-26). Sections 1–7 below
> describe the foundational v11 substrate. Section 8 documents the v15.x
> additions (Pas 6 RoMR + Pas 7a consolidator pipeline) that wrap the
> substrate. See [paper/D_CORTEX_PAS7A_SEAL.md](paper/D_CORTEX_PAS7A_SEAL.md)
> for the citable seal certificate.

## 1. High-Level Overview

D_Cortex v2.0-alpha (sealed v15.7a) is a dual-agent transformer with explicit memory. The architecture separates three concerns:

1. **Writer agent (encoder)**: processes facts, writes to memory
2. **Reader agent (decoder)**: processes questions, reads from memory, produces output
3. **Shared semantic infrastructure**: guarantees key-query compatibility between agents

Total parameters: 175.81M on a 768-dim, 12-head, 12-layer decoder with 4-layer encoder.

## 2. Module Specifications

### 2.1 SharedAddressEncoder (C_sigma)

**Location**: `dcortex/shared_address.py`

**Purpose**: Produce address codes from raw token+position embeddings. Same function applied to fact tokens (for key generation) and question tokens (for query generation). This is the structural foundation of address compatibility.

**Architecture**:
```
Input: embeddings [B, T, D] (D = hidden_dim = 768)

LayerNorm
  |
  v
MultiheadAttention (self-attention)  [H = max(4, n_enc_heads // 2)]
  |
  + Residual
  |
  v
LayerNorm
  |
  v
MultiheadAttention (learned query pool -> attends to sequence)
  |
  v
LayerNorm (output)

Output: address_code [B, D]
```

**Parameters**: approximately 1M (1 self-attention + 1 pool attention + 3 LayerNorms + 1 learned query parameter)

**Key property**: For two token sequences sharing content tokens, the output vectors have measurable cosine similarity (+0.03 at initialization without training, growing with training). This is not achieved by any deeper architecture; the attention pool over shared embeddings produces shared address features structurally.

### 2.2 QueryEngine (K_phi)

**Location**: `dcortex/memory/query.py`

**Purpose**: Project address codes into three key/query spaces (entity, relation, type).

**Architecture**:
```
Input: addr_code [B, D]

LayerNorm
  |
  +-> Linear(D, d_ent = 128)  -> entity key/query
  |
  +-> Linear(D, d_rel =  64)  -> relation key/query
  |
  +-> Linear(D, d_typ =  64)  -> type key/query

Output: (k_ent [B, 128], k_rel [B, 64], k_typ [B, 64])
```

**Parameters**: approximately 0.20M

**Sharing**: Single instance shared between writer and reader. Writer passes `addr_code_from_fact`; reader passes `addr_code_from_question`. Since both pass through the same `shared_address_encoder` first, fact key and question query live in the exact same geometric space.

### 2.3 MemoryBank

**Location**: `dcortex/memory/banks.py`

**Purpose**: Store (key, value) pairs with metadata. Each bank has a fixed capacity.

**Buffers (persistent, no gradient)**:
- `k_ent [capacity, 128]`: entity keys
- `k_rel [capacity, 64]`: relation keys
- `k_typ [capacity, 64]`: type keys
- `values [capacity, 768]`: stored values
- `occupied [capacity]`: bool, slot in use
- `usage [capacity]`: float, LRU counter
- `last_write_step [capacity]`: int, timestamp

**Differentiable overlay**: A dictionary `{slot_idx: {value, k_ent, k_rel, k_typ}}` populated by the writer with gradient-carrying tensors. Readers use `get_diff_*()` which overlays differentiable copies on top of buffer values, enabling end-to-end gradient flow from decoder back to encoder through stored content.

**Capacity by bank**:
- `state_memory`: 64
- `episode_obj_memory`: 128
- `conflict_memory`: 32
- `archive_memory`: 512
- `working_memory`: 16 (primary during v11 training)

### 2.4 MemoryWriter

**Location**: `dcortex/memory/writer.py`

**Purpose**: Route fact writes to banks. Produce value with lexical binding.

**Inputs**:
- `h_pool`: contextual encoder pooled output [B, D]
- `addr_code`: address code from shared encoder [B, D]
- `answer_emb`: embedding of answer token (for lexical binding) [B, D] or None
- `lexical_alpha`: weight on lexical component, default 0.9
- `force_bank`: name of bank to force, default None

**Key generation** (uses shared query engine):
```
k_ent, k_rel, k_typ = shared_query_engine(addr_code)
```

**Value generation**:
```
value_contextual = value_head(LayerNorm(h_pool))
  where value_head = Linear(D, D) -> GELU -> Linear(D, D)

if answer_emb is provided:
    value_lexical = lexical_W_v(answer_emb)
    value = alpha * value_lexical + (1 - alpha) * value_contextual
else:
    value = value_contextual
```

**Bank selection**:
```
gate = Softmax(Linear(D, 6)(LayerNorm(h_pool)))   # 5 banks + skip
if force_bank is not None:
    choice = index_of(force_bank)
else:
    choice = argmax(gate[:5])  # excluding skip
```

**Writing**:
- Calls `MemoryUpdater.update(bank, value_detached, k_ent_detached, k_rel_detached, k_typ_detached, step)`
- Writes differentiable versions to overlay: `bank.set_overlay(slot, value, k_ent, k_rel, k_typ)`

**Returns**: `{gate_probs, value, k_ent, k_rel, k_typ, slot_writes}` where `slot_writes` is list of `(bank_name, slot_idx)` per batch element.

### 2.5 MemoryUpdater

**Location**: `dcortex/memory/updater.py`

**Purpose**: Decide where to place a new write within a bank: allocate fresh slot, update existing matching slot, or evict LRU.

**Algorithm**:
```
1. Compute similarity of new keys against all occupied slots.
2. Find best match (slot with highest similarity).
3. If bank has free slot AND best_sim < theta_match (0.85):
     Allocate new slot.
4. If best_sim >= theta_match:
     Update existing slot via EMA: 
       new_value = (1 - ema_alpha) * old + ema_alpha * new
     with ema_alpha = 0.9 in v11 (replaces, not blends).
5. Otherwise (bank full, no match):
     Evict LRU slot, write new values.
```

**Conflict detection** (separate method): checks if incoming keys match but values differ substantially.

### 2.6 SemanticReader

**Location**: `dcortex/memory/readers.py`

**Purpose**: Read from a single bank given queries.

**Algorithm**:
```
If bank is empty:
    return zeros [B, D]

Normalize queries and keys:
    q_ent_n = normalize(q_ent)
    k_ent_n = normalize(bank.get_diff_k_ent())   # overlay-aware
    ... (same for rel, typ)

Compute weighted similarity:
    sim = w_ent * (q_ent_n @ k_ent_n.T) 
        + w_rel * (q_rel_n @ k_rel_n.T)
        + w_typ * (q_typ_n @ k_typ_n.T)

Mask unoccupied slots:
    sim = sim.masked_fill(~occupied, -inf)

Apply temperature:
    attn = softmax(sim * 20.0)   # TEMPERATURE = 20 in v10+

Read values:
    r = attn @ bank.get_diff_values()

Return: r [B, D]
```

**Critical parameters**:
- `query_weights = (1.0, 0.0, 0.0)` in v10+: only entity key matters (matches L_sel supervision)
- Temperature = 20: sharpens softmax from [0.59, 0.14, 0.14, 0.14] to [~1, ~0, ~0, ~0] for clear matches

### 2.7 MemoryReadFusion

**Location**: `dcortex/memory/readers.py`

**Purpose**: Combine reader outputs from all 5 banks into a single "memory tokens" tensor for decoder cross-attention.

**Architecture**:
```
Input: (r_state, r_episode, r_conflict, r_archive, r_working)  each [B, D]

For each reader output:
    proj = Linear(D, D)(r)

Stack: [B, 5, D]

LayerNorm across the stream dimension.

Output: memory_tokens [B, 5, D]
```

**Important note**: In v10+, the auxiliary emission path uses `retrieved_value = r_state + r_episode + ... + r_working` (sum of raw reader outputs), NOT `memory_tokens.sum()` or `memory_tokens.mean()`. This bypasses the fusion projection biases that polluted signal when streams were zero.

### 2.8 AuxAnswerHead

**Location**: `dcortex/aux_modules.py`

**Purpose**: Direct projection from retrieved value to vocabulary logits. Primary emission path in validated configuration.

**Architecture**:
```
Input: retrieved_value [B, D]

LayerNorm
  |
  v
Linear(D, D)
  |
  v
GELU
  |
  v
Linear(D, D)
  |
  v
Tied projection to shared_token_emb.T  (output [B, V])
```

**Tying**: Output uses `retrieved_value @ shared_token_emb.weight.T`, which aligns the emission space with the input embedding space. Since values are lexically bound (0.9 * E(ans) + 0.1 * context), the argmax concentrates on the correct answer token after minimal training.

### 2.9 DCortexV2Model

**Location**: `dcortex/model.py`

Top-level class with methods:
- `encode(input_ids, answer_token_id=None, lexical_alpha=0.9, force_bank=None)`: run encoder, write to memory
- `decode(input_ids, attention_mask=None, force_attend=False, return_retrieved=False)`: run decoder, read memory, produce logits. With `return_retrieved=True`, also returns retrieved_value.
- `reset_memory()`: clear all banks
- `begin_episode()`: reset episode-scoped state (SSM)
- `clear_overlays()`: clear differentiable overlays after backward pass
- `memory_snapshot()`: diagnostic snapshot of bank occupancy

Internal state:
- `step_counter`: global step for timestamping writes
- `_enc_aux`: cache of encoder auxiliary outputs for training losses

## 3. Data Flow

### 3.1 Write Path (encode)

```
input_ids (fact tokens)
   |
   v
shared_token_emb + shared_pos_emb
   |
   v
emb_raw [B, T, D]
   |
   +-----> shared_address_encoder ----> addr_code [B, D]
   |                                       |
   v                                       v
encoder_blocks (4 layers)           shared_query_engine
   |                                       |
   v                                       v
h_pool [B, D]                     (k_ent, k_rel, k_typ)
   |                                       |
   +-----> answer_token_id -> shared_token_emb -> answer_emb
   |                                                   |
   v                                                   v
   +----------> MemoryWriter <--------------------------+
                    |
                    v
           MemoryUpdater (allocates slot)
                    |
                    v
           Bank (value + keys stored, overlay set with gradient)
```

### 3.2 Read Path (decode)

```
input_ids (question tokens)
   |
   v
shared_token_emb + shared_pos_emb
   |
   v
emb_raw [B, T, D]
   |
   +-----> shared_address_encoder ----> addr_code [B, D]
   |                                       |
   v                                       v
decoder_standard_blocks (8)       shared_query_engine
   |                                       |
   v                                       v
h [B, T, D]                       (q_ent, q_rel, q_typ)
                                           |
                              +------------+------------+
                              v                         v
                     SemanticReader x 5          (per bank)
                              |
                              v
               (r_state, r_ep, r_conf, r_arch, r_work) each [B, D]
                              |
              +---------------+-------------------+
              |                                   |
              v                                   v
    retrieved_value = sum(*)          MemoryReadFusion (projections)
              |                                   |
              v                                   v
    AuxAnswerHead                        memory_tokens [B, 5, D]
              |                                   |
              v                                   v
    aux_logits [B, V]                decoder_fusion_blocks (4)
    (PRIMARY EMISSION)                           |
                                                  v
                                         dec_lm_head
                                                  |
                                                  v
                                         logits [B, T, V]
                                         (LM emission)
```

## 4. Configuration Reference

### 4.1 Architecture (config.py)

```python
@dataclass
class DCortexConfig:
    # Backbone
    hidden_dim: int = 768
    n_enc_heads: int = 12        # encoder heads
    n_dec_heads: int = 12        # decoder heads
    n_enc_layers: int = 4        # encoder layers (writer)
    n_dec_layers: int = 12       # decoder layers (reader; 8 std + 4 fusion)
    n_fusion_blocks: int = 4     # how many of dec_layers are fusion
    enc_ff_dim: int = 3072
    dec_ff_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 1024
    vocab_size: int = 50257      # GPT-2
    
    # Memory banks
    state_capacity: int = 64
    episode_obj_capacity: int = 128
    conflict_capacity: int = 32
    archive_capacity: int = 512
    working_capacity: int = 16
    
    # Keys
    d_ent: int = 128
    d_rel: int = 64
    d_typ: int = 64
    query_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # v10+
    
    # Updater
    theta_match: float = 0.85    # cosine threshold for key match
    theta_conflict: float = 0.3   # conflict detection
    ema_alpha: float = 0.9        # v11+: 90% new value on update
    
    # SSM (episode state)
    ssm_hidden_dim: int = 256
    
    # Init
    init_std: float = 0.02
```

### 4.2 Training (per script)

See `colab/step2_training_v6.py` (v10) and `colab/step2_training_v11.py` (v11) for `TrainConfig` dataclasses.

## 5. Critical Design Decisions

### 5.1 Why Shared Infrastructure?

Without sharing, writer keys and reader queries live in different spaces. Training must align them through gradient descent over many episodes. This is slow and brittle. Sharing makes alignment structural: at initialization, sequences with shared tokens produce similar keys/queries by construction.

### 5.2 Why Temperature 20 in Reader?

Cosine similarities live in [-1, 1]. Softmax over [1, 0, 0, 0] gives [0.59, 0.14, 0.14, 0.14]. Even optimal key matching leaves 41% attention mass on wrong slots. Temperature 20 scales the range to [-20, 20], producing near-hard attention on best match.

### 5.3 Why query_weights (1, 0, 0)?

L_sel supervision only aligns entity keys. Including unsupervised relation and type similarities in the reader's weighted sum adds noise that the supervision cannot counteract. Setting them to zero aligns reader behavior with what supervision can actually train.

### 5.4 Why ema_alpha = 0.9?

Updates should replace, not blend. With 0.3 (original), stored value after one update is 70% old + 30% new, which the decoder reads as ambiguous. With 0.9, it's 10% old + 90% new, and chain stability is mathematical: after N updates, initial influence is 0.1^N, negligible after 2-3 updates.

### 5.5 Why force_bank='working' During Validation?

Without it, writer scatters facts across banks, producing one-slot-per-bank where softmax attention is always 1.0 regardless of query. All bank contributions sum regardless of query, making retrieval query-independent. force_bank puts all facts in one pool where attention discrimination becomes meaningful. This is a validation crutch, not a long-term architectural choice; learning autonomous bank selection is future work.

### 5.6 Why Lexical Binding?

Without it, stored values are abstract pools containing context but not decoded tokens. Even with correct retrieval, the decoder cannot extract the answer token reliably. Lexical binding makes stored values directly decodable by tying the auxiliary head to the shared token embedding.

## 6. Known Limitations

1. **Working bank capacity**: 16 slots limits number of facts per episode
2. **Other banks dormant**: state, episode_obj, conflict, archive not functionally differentiated in current training
3. **No consolidation**: LRU eviction exists but no migration between banks
4. **Fixed templates**: episodes use rigid "The X is Y" format
5. **All entities seen in training**: no true held-out generalization validated
6. **Aux head dominant**: LM path functional but secondary for emission

## 7. Extension Points

To extend D_Cortex for new tasks:

1. **Different answer types**: modify `answer_token_id` generation (currently first token of " answer"). For multi-token answers, consider span-based binding.
2. **Different question formats**: the current `encode_text(prompt)` handles any GPT-2 tokenizable string, but shared_address_encoder is trained on fact templates. Paraphrases require v13 retraining.
3. **New banks**: add capacity in config, add corresponding reader to `DCortexV2Model.decode()`, add to `MemoryReadFusion`, add to `retrieved_value = sum(...)`.
4. **Different emission paths**: implement alternative heads; `AuxAnswerHead` is one example. Any module taking [B, D] to [B, V] can be used.
5. **Natural corpora**: current training mixes structural + TinyStories. Replace TinyStories with target domain for domain adaptation.

---

## 8. v15.x Layer — Memory That Operates on Its Own History

The architecture above (sections 1–7) describes the **v11 substrate**: dual-agent
transformer with explicit memory banks, validated for memory-conditioned token
emission (94.4% on 4-fact retrieval, 100% on 8-step update chains).

The **v15.x layer** wraps this substrate with three additional sub-systems,
each sealed in a separate development step:

### 8.1 v15.6 Pas 6 — Role-of-Modifier Resolver (RoMR)

**Location**: `steps/12_v15_6_pas6_romr/code.py` (sealed bundle).

**Problem**: in the v11 substrate, the parser conflated `ENTITY_MODIFIER` ("a
red dragon") with `ATTRIBUTE_VALUE` ("the dragon is red"). On the F2 holdout
(multiword entities), `attr_write_failure` was 21.8 %.

**Solution**: token-level role classification BEFORE the verifier runs.
`RoleOfModifierResolver` labels each value candidate as one of
`ENTITY_MODIFIER` / `ATTRIBUTE_VALUE` / `UNCERTAIN` based on:
- position relative to the noun-phrase span
- presence of a copula in `V15_6_PAS6_COPULAS` (33 linking verbs:
  *is, was, became, seemed, appeared, stood, looked, felt, …*)
- packet-level `REAL_CONFLICT` flag when the same attribute family has
  ≥ 2 distinct values ("the small horse is huge")

`CommitArbiterPas6` extends `CommitArbiterPas3`. RoMR runs after `v15_4`
parser, before verifier, on a shallow copy of the packet. The raw packet is
preserved for audit; only `value_candidates` are filtered.

**Result**: F2 safe_resolution 0.782 → **0.952**, attr_write_fail 21.8 % →
**0.000 %**, all 7 Pas 6 acceptance gates green. S5/S6 honesty 1.000
preserved. Trusted regression byte-identical.

### 8.2 v15.7a Pas 7a — Consolidator Pipeline at end_episode

**Location**: `steps/13_v15_7a_consolidation/code.py` (sealed monolith).

**Problem**: until Pas 6, memory was a passive store. Provisional entries
accumulated indefinitely. Committed values were never demoted, even under
strong contradicting evidence. There was no mechanism by which memory could
operate on its own history.

**Solution**: a synchronous consolidator pipeline at `end_episode`, executed
**after** the existing Pas 2/6 finalize:

```
end_episode():
    [existing Pas 2/6 finalize — unchanged]
    consolidator.reconcile(...)   # collapse exact (slot,value,episode) duplicates
    consolidator.prune(...)       # drop provisional stale beyond K_prune_stale
    consolidator.retrograde(...)  # demote committed slot if M challenger episodes met
    consolidator.promote(...)     # elevate provisional value if N confirms + age >= K_age
```

#### 8.2.1 Derivation Layer (D.3)

Pure functions over `List[ProvisionalEntry]`:

| Function | Returns |
|---|---|
| `_v15_7a_confirmation_episodes(entries, ent, attr, value_idx)` | distinct `episode_id`s where `(slot, value)` was deposited |
| `_v15_7a_last_activity_episode(entries, ent, attr)` | max `episode_id` for any value of this slot |
| `_v15_7a_first_seen_episode(entries, ent, attr, value_idx)` | min `episode_id` for `(slot, value)` |
| `_v15_7a_distinct_values_for_slot(entries, ent, attr)` | set of `value_idx` observed |

Predicates built on top:

| Predicate | Truth condition |
|---|---|
| `is_promote_eligible(entries, ent, attr, v, current_ep)` | `\|confirms\| ≥ N=2` AND `current_ep − first_seen ≥ K_age=2` |
| `is_retrograde_eligible(entries, ent, attr, committed_v)` | exists `v ≠ committed_v` with `\|confirms\| ≥ M=2`; tiebreak: most confirms, then smallest `value_idx` |
| `is_stale_for_prune(entries, ent, attr, current_ep)` | `current_ep − last_activity ≥ K_stale=3` |

#### 8.2.2 Atomic Operations

Each operation emits a `V15_7a_ConsolidationRecord` (audit-trail dataclass:
`episode_id`, `operation`, `entity_id`, `attr_type`, `value_idx`, `reason`,
`state_before`, `state_after`).

| Op | Mutates | Counting | Provisional handling |
|---|---|---|---|
| **reconcile** (D.4) | provisional only | 1 per fired slot (dedup duplicates by `(value, episode)`) | collapse, write_step tiebreak (earliest wins) |
| **prune** (D.5) | provisional only | 1 per pruned entry (matches L5 `PRUNE: 2`) | drop entire stale slot |
| **retrograde** (D.6) | bank + stability_index | 1 per demoted slot | NOT touched in v1 (challenger remains for D.7) |
| **promote** (D.7) | bank + stability_index + provisional | 1 per promoted slot | clear `(slot, promoted_value)` only |

#### 8.2.3 Bank-State Policy for Promote (No Transitive Demote)

Promote is bound by an explicit four-case policy:

| Bank state for `(ent, attr)` | Action |
|---|---|
| entity not in bank | `PROMOTE_SKIPPED` ("v1 consolidator does not allocate") |
| `attr_slot.present == False` (post-retrograde or never-committed) | promote: in-place mutation, mark stability |
| `present == True` and `value_idx == promote_value` | idempotent finalize (re-mark stability, clear provisional) |
| `present == True` and `value_idx != promote_value` | `PROMOTE_SKIPPED` ("D.6 didn't demote it ⇒ not eligible for transitive overwrite") |

#### 8.2.4 Intra-Pas Exclusion

A slot retrograded in the **same** `end_episode` is excluded from the promote
loop in that same episode. Implementation: scan the audit log for
`RETROGRADE` records with `episode_id == current_episode`, build the exclusion
set, skip those slots. This produces the L1 1-episode delay
(retrograde at `end_ep3`, promote at `end_ep4`, never at `end_ep3`).

#### 8.2.5 Wiring (D.8)

`CommitArbiterPas7a(CommitArbiterPas6)` overrides `end_episode` only:

```python
def end_episode(self, ent_fn, cls_fn, val_fn):
    ep_id = self.episode_buffer.episode_id
    finalize = super().end_episode(ent_fn, cls_fn, val_fn)
    ops = _v15_7a_run_consolidator_pipeline(
        self.provisional_memory, self.bank, self.stability_index,
        current_episode=finalize.episode_id,
        audit=self.consolidation_audit_log,
    )
    self.last_consolidator_ops = ops
    return finalize
```

Pas 6 in-episode behavior (`write_fact`, RoMR filter, dual conflict rule,
cross-episode challenger detection) is **byte-identical** under Pas 7a.
Read path (`ReadArbiter`) is untouched. Query path is untouched.

#### 8.2.6 Acceptance Gates (D.9)

The full evaluator `v15_7a_run_full_eval_d9(base_model, v15_1_memory)` runs
two phases and verifies 10 gates. All passed (2026-04-26, A100,
n_per_l_family=20, seed=20261103, 100 sequences total):

| Gate | Threshold | Result |
|---|---|---:|
| 0 | trusted regression byte-identical | PASS |
| 1 | wrong_commit ≤ 0.02 across F1-F5 | PASS (0.000) |
| 2 | F2 safe_resolution ≥ 0.95 | PASS (0.952) |
| 3 | false_promote_rate = 0 | PASS (0/100) |
| 4 | false_retrograde_rate = 0 | PASS (0/100) |
| 5 | L1 promote_rate ≥ 0.95 | PASS (1.000) |
| 6 | L2 retrograde_rate ≥ 0.90 | PASS (1.000) |
| 7 | L3 false_retrograde = 0 on completions | PASS (0/20) |
| 8 | L4 promote_count = 0 (anti-inflation) | PASS (0/20) |
| 9 | L5 prune_count ≥ 1 per stale trial | PASS (2/trial) |

### 8.3 What Pas 7a Adds to the Architecture (in plain terms)

| Before Pas 7a | After Pas 7a |
|---|---|
| Memory was a passive store. | Memory operates on its own history. |
| Committed values were permanent. | **Stable can fall** under M=2 distinct contradicting episodes. |
| Provisional entries accumulated. | **Provisional can rise** with N=2 distinct confirms + K_age=2 maturity. |
| No reflexive structure. | Each consolidator op leaves an audit record with `state_before` / `state_after`. |
| Single-episode tests sufficient. | Longitudinal regime (5 L families × 20 trials) measures cross-episode dynamics. |

### 8.4 What Pas 7a Does NOT Do

- Does not contaminate Pas 6 critical path (Gate 0 byte-identical).
- Does not touch the read path or the query path.
- Does not allocate new entities (consolidator is bank-mutator only on already-allocated entities).
- Does not silently overwrite stable bank values (no transitive demote in v1).
- Does not regenerate value embeddings on promote (`value_emb` left as-is).
- Does not run as a background process — strictly synchronous at `end_episode`.

---

**End of Architecture Documentation**
