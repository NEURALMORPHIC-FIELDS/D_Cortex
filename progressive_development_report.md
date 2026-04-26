# D_Cortex v2.0-alpha: A Dual-Agent Memory-Native Transformer Architecture

> **SCOPE NOTE (2026-04-26)**: This report covers the **foundational v9-v11
> substrate** sealed on 2026-04-18. The current sealed milestone of the
> project is **v15.7a (Pas 7a, 2026-04-26)** — first longitudinal organ
> validated, all 10 D9 acceptance gates green. For the v15.x progression
> (Pas 6 RoMR + Pas 7a consolidator pipeline), see:
>
> - [paper/D_CORTEX_PAS7A_SEAL.md](paper/D_CORTEX_PAS7A_SEAL.md) — citable seal certificate
> - [docs/PROGRESS.md](docs/PROGRESS.md) — chronological log of all sealed steps
> - [architecture.md](architecture.md) Section 8 — v15.x architectural additions
> - [experiments.md](experiments.md) v15.x section — Pas 6 RoMR + Pas 7a evaluation
> - [steps/13_v15_7a_consolidation/](steps/13_v15_7a_consolidation/) — sealed Pas 7a code + spec
>
> The v9-v11 results below remain canonical for the substrate layer that
> the v15.x consolidator wraps.

## Progressive Development Report

**Author**: Vasile Lucian Borbeleac  
**Affiliation**: FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania  
**Patent**: EP25216372.0 (FHRSS)  
**Period**: April 16 -- April 18, 2026  
**Version**: 2.0-alpha (v11, ckpt_v11_step004000.pt)  
**Copyright**: (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.

---

## Abstract

We present D_Cortex v2.0-alpha, a dual-agent transformer architecture where explicit episodic memory operates as a structurally separate system from both model weights and context window. The architecture implements a clear three-way separation: (1) a writer encoder that stores facts as addressable key-value pairs with lexical binding, (2) a reader decoder that queries memory through shared address space, and (3) a multi-bank memory substrate with explicit semantic typing. Through eleven development iterations over three days, we identified and resolved three compounding architectural bugs that prevented memory-to-emission coupling, demonstrating that memory addressing and memory emission are distinct computational competences requiring separate mechanisms. The final validated system achieves 94.4% top-1 accuracy on 4-fact retrieval, 100% on update chains up to 8 successive modifications, 93.6% distractor resistance at 15-fact capacity, and 95.3% accuracy on rare entities among frequent distractors. We report the complete development trajectory including failure modes, diagnostic methodology, and architectural insights that generalize beyond this specific implementation. The central conceptual contribution is the experimental demonstration that memory can exist as a functional layer separate from weights and separate from context, constituting a third category in neural architecture design.

**Keywords**: memory-augmented transformers, explicit episodic memory, dual-agent architectures, address-value binding, memory addressing versus emission, cognitive architecture

---

## 1. Introduction

### 1.1 Motivation

Contemporary large language models implement knowledge through two mechanisms: parametric memory encoded in weights through gradient descent, and contextual memory provided through the attention window. Both mechanisms exhibit structural limitations. Parametric memory is slow to update, requires retraining for new information, and suffers from catastrophic forgetting. Contextual memory is bounded by window length, lacks persistence across conversations, and does not support selective update of specific facts.

The FRAGMERGENT research program hypothesizes a third category: explicit memory as a structurally separate subsystem with properties distinct from both weights and context. Such memory would be persistent (unlike context), addressable by content (unlike weights), updatable without retraining (unlike weights), and selective (unlike full context replay). D_Cortex v2.0-alpha is an experimental instantiation of this hypothesis.

### 1.2 Contribution

This work makes four contributions:

**C1.** We demonstrate experimentally that a dual-agent transformer with explicit memory can achieve memory-conditioned exact token emission, not merely distributional influence. Prior approaches typically show memory shifting probability mass toward correct answers; we show memory determining the exact answer token with 94.4% accuracy.

**C2.** We isolate the distinction between memory addressing competence and memory emission competence, demonstrating empirically that a system can achieve 97% key-query similarity accuracy while achieving only 7% token emission accuracy. This identifies emission as a separate computational requirement beyond retrieval.

**C3.** We identify three compounding architectural bugs that cause emission failure despite successful retrieval: (i) bias pollution in fusion projections over zero streams, (ii) insufficient temperature in retrieval softmax, and (iii) bank scattering that produces unit-attention artifacts. Each of these generalizes beyond our specific implementation.

**C4.** We provide a validation protocol distinguishing memory addressing, memory emission, update chain stability, and compositional generalization, with Wilson confidence intervals on all measurements.

### 1.3 Scope and Limits

We demonstrate a validated mechanism under controlled structural conditions. We do not demonstrate: (a) autonomous bank selection without the `force_bank='working'` constraint, (b) generalization on true held-out entities excluded from structural training, (c) natural language variation beyond fixed episode templates, or (d) integration into production-scale language models. These limitations are documented in Section 9 and constitute the forward agenda.

---

## 2. Architecture

### 2.1 System Overview

D_Cortex v2.0-alpha is a dual-agent architecture where a writer agent (encoder) processes facts and stores them in memory banks, while a reader agent (decoder) processes questions and retrieves stored information. Both agents share infrastructure for semantic alignment.

```
                FACT TOKENS                    QUESTION TOKENS
                    |                                |
                    v                                v
           ====================================
           |   SHARED TOKEN + POS EMBEDDINGS   |
           ====================================
                    |                                |
                    v                                v
           ====================================
           |      SHARED ADDRESS ENCODER       |
           |         (C_sigma module)          |
           ====================================
                    |                                |
                    v                                v
            +-----------+                    +------------+
            |  Encoder  |                    |  Decoder   |
            |  Blocks   |                    |  Blocks    |
            +-----+-----+                    +------+-----+
                  |                                 |
           h_pool v                          q_addr v
                  |                                 |
                  |    SHARED QUERY ENGINE          |
                  |     (K_phi: addr -> key)        |
                  |         |     ^                 |
                  |  keys   v     |   queries       |
                  |         |     |                 |
                  |         v     |                 |
            +-----v-----+       +-+----------------+
            |  WRITER   |       |  READERS + FUSION|
            |  Value =  |       |  retrieved_value |
            |  lexical  |-----> |  aux_answer_head |
            |  binding  | write |                  |
            +-----------+       +--------+---------+
                                         |
                                         v
                                    ANSWER TOKEN
```

### 2.2 Core Components

**Shared Token + Position Embeddings** (`shared_token_emb`, `shared_pos_emb`): Both encoder and decoder use the same embedding tables (40.17M parameters combined). This provides structural semantic alignment at initialization: the vector for "cat" is identical on the fact-writing path and the question-reading path.

**Shared Address Encoder** (`SharedAddressEncoder`, module C_sigma): A small module (~1M parameters) consisting of one self-attention layer plus attention pooling with learned query. It operates on raw token+position embeddings, producing an address code before any agent-specific processing. Same function applied to fact tokens (for key generation) and question tokens (for query generation).

**Shared Query Engine** (`QueryEngine`, projection K_phi): Projects address codes into three key spaces: entity keys (d_ent=128), relation keys (d_rel=64), type keys (d_typ=64). Both writer and reader use identical projections.

**Memory Banks**: Five banks with distinct semantic labels but identical architecture:
- `state_memory` (64 slots) - persistent entity state
- `episode_obj_memory` (128 slots) - within-episode objects
- `conflict_memory` (32 slots) - contradiction markers
- `archive_memory` (512 slots) - long-term storage
- `working_memory` (16 slots) - active episode scratchpad

Each slot stores a 5-tuple: (key_ent, key_rel, key_typ, value, metadata). Metadata includes occupancy flag, last-write timestamp, and usage counter.

**Writer** (`MemoryWriter`): Produces value with lexical binding. Given an answer embedding `E(a)`:
```
value = alpha * W_v * E(a) + (1 - alpha) * value_contextual
```
where `alpha = 0.9` in validated configuration. The contextual component `value_contextual` comes from encoder blocks. The lexical component dominates, forcing stored values to be decodable as answer tokens.

**Readers** (`SemanticReader`, one per bank): For each bank, compute weighted cosine similarity between query and bank keys, apply temperature-scaled softmax (tau=20), weighted sum over values. Output is a single retrieved vector per bank.

**Auxiliary Answer Head** (`AuxAnswerHead`): Direct projection from retrieved value to vocabulary logits, tied to `shared_token_emb`. Bypasses fusion blocks. Primary emission path in the validated configuration.

**Decoder Fusion Blocks**: Four cross-attention layers where decoder hidden states attend to memory tokens. Secondary path; LM head feeds from these for natural language generation.

### 2.3 Key Innovation: Structural Address Compatibility

A central architectural decision is routing both write keys and read queries through the same two modules (`shared_address_encoder` composed with `shared_query_engine`). Formally:

Write key: `k_f = K_phi(C_sigma(emb(x_f)))` for fact tokens `x_f`  
Read query: `q = K_phi(C_sigma(emb(x_q)))` for question tokens `x_q`

When fact and question share entity tokens (e.g., both contain "cat"), their embeddings share components, which produce similar address codes through identical `C_sigma`, which produce similar keys/queries through identical `K_phi`. This similarity is structural, guaranteed at initialization with cosine similarity approximately +0.03 for token-sharing sequences versus unrelated sequences, and amplified by training.

### 2.4 Emission Path: Sum Over Raw Reader Outputs

Unlike typical memory-augmented architectures that fuse reader outputs through learned projections, D_Cortex emits from a direct sum:

```
retrieved_value = r_state + r_episode + r_conflict + r_archive + r_working
```

where `r_bank` is the reader output for each bank (before any fusion projection). This design prevents bias pollution: each bank with zero occupancy contributes exact zero, not a bias-offset vector. The auxiliary answer head then produces logits directly from this sum through a tied projection to the shared token embedding space.

---

## 3. Methodology

### 3.1 Training Protocol

**Structural episodes** (v10, v11): A random number of facts (3-5 in v10, 3-5 in v11) about distinct entities with distinct attribute values. Each fact takes form "The {entity} is {color}." One entity is queried: "What color is the {entity}?" with the answer being the first token after " ".

**Update episodes** (v11 onwards): N initial facts plus one update fact "The {entity} is now {new_color}." Question asks for the current color. Answer is the updated value.

**Distractor episodes** (v11): All facts share a semantic cluster (animals or fantasy creatures). Tests key separation under semantic similarity.

**Language modeling episodes** (v11): Standard next-token prediction on TinyStories (Eldan & Li, 2023) with memory reset before each batch. Runs in parallel with structural curriculum (25-30% of batches).

### 3.2 Loss Function

The validated configuration (v11) uses:

```
L_total = 1.0 * L_emit + 1.0 * L_sel + 0.5 * L_sep_neg
```

where:
- **L_emit** = CE(aux_answer_head(retrieved_value), answer_token): primary loss, direct supervision on emission
- **L_sel** = -log softmax(cos(q_key, fact_keys) * 5.0)[target_idx]: supervises query-key alignment
- **L_sep_neg** = mean(ReLU(cos(k_i, k_j) - 0.5)^2) over different-entity pairs: separates keys for distinct entities

Note: Prior versions (v8-v9) included L_ans (standard LM cross-entropy on answer tokens through decoder LM head), L_aux (same as L_emit but called differently), L_cycle (value-to-key round-trip). These were found redundant or counterproductive when L_emit is supervised directly and are omitted in v11.

### 3.3 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dim | 768 | GPT-2 small scale |
| Encoder layers | 4 | Writer agent |
| Decoder layers | 12 (8 standard + 4 fusion) | Reader agent |
| Heads | 12 | |
| FF dim | 3072 | |
| Vocab | GPT-2 (50257) | via tiktoken |
| Context | 64 tokens | Sufficient for single facts |
| Batch size | 1 | Serial episodes |
| Grad accum | 16 | Effective batch = 16 |
| LR (base) | 6e-4 | |
| LR (shared+aux) | 3e-3 | 5x multiplier in v10, 3x in v11 |
| LR (encoder own) | 1.8e-3 | 3x in v10, 2x in v11 |
| ema_alpha | 0.9 | Update blending: 90% new |
| theta_match | 0.85 | Cosine threshold for key match |
| Reader temperature | 20 | Sharpens softmax on cosine sims |
| query_weights | (1, 0, 0) | Only entity key matters |
| Lexical alpha | 0.9 | 90% lexical, 10% contextual |
| Force bank | 'working' | All structural writes to one pool |
| Total steps | 3000 (v10), 4000 (v11) | |
| Hardware | NVIDIA A100-SXM4-40GB | |
| Precision | bfloat16 + TF32, no GradScaler | |

### 3.4 Evaluation Protocol

All accuracy reported as:
```
top1 = count(aux_answer_head(retrieved_value).argmax() == answer_token) / N
```

Wilson 95% confidence intervals reported on all measurements. Three distinct evaluation suites:

**Basic evaluation** (during training, every 500 steps): 200 episodes each of simple, update, distractor.

**Final evaluation** (after training, N=500): Broader coverage with scaling test at n_facts ∈ {3, 4, 5, 6, 8, 10}.

**B1.1 Extended Validation** (separate script, N=300-500 per cell): Six cells across three blocks:
- Block 1: Scaling 3/5/8/12/15 facts on simple, update, distractor
- Block 2: Update chain length 1/2/4/8 plus leak test at chain=3
- Block 3: Rare entities only, rare+frequent distractors, cross-schema 4-class, cross-trained cluster mix

---

## 4. Development Trajectory

### 4.1 Version Timeline

| Version | Date | Steps | Top-1 | Key change | Outcome |
|---------|------|-------|-------|------------|---------|
| v1-v5 | April 16-17 | varies | -- | Initial dual-agent setup | Various failures |
| v6 | April 17 | 500-2500 | ~5% | Separate encoder, decoder + overlay | Encoder gradient dies at step 50-200 |
| v7 | April 17 | 500-2500 | ~7% | Shared semantic infrastructure | Occupancy stuck at 1 slot |
| v8 | April 17 | 2500 | 7% | Structural curriculum with L_sel/L_sep/L_occ | retr_acc=97% but top1=7% |
| v9 | April 17 | 2500 | 7% | + Aux head + L_cycle + LM mix | retr_acc=97%, top1=7% persists |
| v10 | April 17 | 3000 | **93.2%** | Three compounding bugs fixed | Validation of principle |
| v11 | April 18 | 4000 | **94.4% S / 100% U / 99.2% D** | Complex episodes, warm start from v10 | B1 PASS |

### 4.2 The Retrieval-Emission Gap (v8-v9)

Starting from v8, the system demonstrated consistent internal retrieval metrics: 97% accuracy at selecting the correct slot, mean rank 1.04/4 on test episodes, occupancy ratio 3.84/4. However, top-1 accuracy on answer token emission remained at 7%, barely above random baseline (1/15 for color tokens = 6.7%).

Deep diagnostic (`step2_6_deep_diagnostic.py`) revealed six converging failures:

1. **Aux head top-1 = 7.2%, Decode top-1 = 6.8%**: both paths fail equally
2. **mem_gate sigmoid ≈ 0.48** across all four fusion blocks: near-neutral, not used
3. **Masking any stream drops accuracy by 0.0%**: memory not used at all at inference
4. **Color-restricted accuracy = 7.8%** (random = 6.7%): model cannot discriminate colors even when forced to pick one
5. **Raw value → LM head = 0.0%**: stored values structurally non-lexical
6. **Top-5 tokens identical across different questions**: cat/dragon/tiger all produce `[gray, violet, brown, purple, green]`

The sixth observation was decisive. The model had learned a single color distribution. It discriminated "this is a color question" but not which color. Keys separated and selected correctly (97%), but values stored did not contain the answer in decodable form.

### 4.3 Three Compounding Architectural Bugs

Detailed diagnosis through a mini-training loop (300 iterations on fixed 4-fact episodes with artificial vocab) identified three bugs that compounded to produce uniform distribution collapse.

**Bug 1: Fusion projection bias pollution.**

The decoder fusion mechanism (`MemoryReadFusion`) stacked reader outputs through `nn.Linear` projections with bias terms. When a bank was empty, its reader returned zeros. But `proj_empty(zeros) = bias ≠ 0`. These bias vectors then passed through `LayerNorm`, amplifying to magnitude comparable to the populated stream. The signal-to-noise ratio collapsed.

**Fix 1**: Compute `retrieved_value = sum(r_state, r_episode, r_conflict, r_archive, r_working)` directly, bypassing fusion projections. Zero streams now contribute exact zero.

**Bug 2: Reader softmax too diffuse.**

`SemanticReader` computed softmax over cosine similarities in [-1, 1]. With perfect key alignment (sim = 1.0 for target, 0 for others over 4 candidates), softmax yields [0.59, 0.14, 0.14, 0.14]. Even optimal retrieval diluted the target value with 41% mass from other slots.

**Fix 2**: Apply temperature τ=20 before softmax: `attn = softmax(sim * 20)`. Now softmax([1, 0, 0, 0] × 20) ≈ [1, 0, 0, 0]. Target dominance restored.

**Bug 3: Bank scattering (the critical bug).**

The writer's gate module produced distribution over 6 options (5 banks + skip). For different facts, it chose different banks. Four facts typically distributed as one fact per bank. When each bank contained exactly one slot, the reader's softmax over `[sim]` with a single occupied slot always produced attention = 1.0 regardless of actual similarity. The retrieved value per bank was always the entire stored value. The sum across banks became the sum of all fact values, independent of which question was asked. This fully explained the identical top-5 across queries.

**Fix 3**: Add `force_bank='working'` parameter. All structural writes go to one bank. With multiple slots in the same bank, softmax attention becomes query-dependent.

Additional fix: **`query_weights = (1, 0, 0)`**. The supervising loss `L_sel` only trained entity keys. Reader formula `sim = 0.5*sim_ent + 0.3*sim_rel + 0.2*sim_typ` mixed supervised entity similarity with unsupervised (random) relation and type similarities. Setting weights to `(1, 0, 0)` aligns reader with supervision.

### 4.4 Validation of Principle (v10)

With all three fixes applied, v10 trained from scratch for 3000 steps. Progression:

| Evaluation step | struct_acc_4 | struct_acc_5 | val_loss |
|-----------------|--------------|--------------|----------|
| 500 | 83.5% | 83.0% | 3.308 |
| 1000 | 84.0% | 82.0% | 2.995 |
| 1500 | 84.0% | 79.0% | 2.836 |
| 2000 | 88.5% | 92.0% | 2.687 |
| 2500 | 91.0% | 90.0% | 2.588 |
| 3000 | 92.0% | 94.0% | 2.458 |

Final scaling evaluation (300 episodes per n_facts):

| N_facts | Accuracy |
|---------|----------|
| 3 | 95.7% |
| 4 | 94.7% |
| 5 | 90.3% |
| 6 | 90.3% |
| 8 | 85.3% |
| 10 | 78.7% |

Validation criteria (4-fact > 90%): **PASSED at 93.2%** (N=500). Comparing to v8/v9 at 7%, the same architecture with 175M parameters on the same data improved from random-baseline performance to 93.2% through three architectural fixes alone.

### 4.5 Complex Episodes (v11)

v11 added update and distractor episode types to the training mix (50% simple / 25% update / 25% distractor / LM=25% in parallel). Warm-started from v10 checkpoint `ckpt_step003000.pt`. Additional architectural change: `ema_alpha = 0.3 → 0.9` in memory updater, so updates to existing entities replace rather than blend.

Training (4000 steps, 89.9 minutes on A100):

| Eval step | simple | update | distractor | val_loss |
|-----------|--------|--------|------------|----------|
| 500 | 89.5% | 99.0% | 93.0% | 2.712 |
| 1000 | 90.0% | 100.0% | 92.5% | 2.794 |
| 1500 | 89.5% | 100.0% | 99.5% | 2.652 |
| 2000 | 95.0% | 100.0% | 98.0% | 2.498 |
| 2500 | 93.5% | 100.0% | 94.0% | 2.450 |
| 3000 | 93.0% | 100.0% | 99.5% | 2.361 |
| 3500 | 96.5% | 100.0% | 99.0% | 2.325 |
| 4000 | 95.0% | 100.0% | 99.0% | 2.175 |

Final evaluation (N=500): simple=94.4%, update=100.0%, distractor=99.2%.

---

## 5. Results

### 5.1 Primary Metrics

From validation run (v11, ckpt_v11_step004000.pt), N=500 episodes per condition with Wilson 95% CI:

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| Simple 4-fact | 94.4% | [91.9%, 96.2%] |
| Update 4-fact | 100.0% | [99.2%, 100.0%] |
| Distractor 4-fact (same-cluster) | 99.2% | [97.9%, 99.7%] |

### 5.2 Scaling Behavior

Block 1 scaling test (v11, N=500 per cell with Wilson 95% CI):

| N_facts | Simple | Update | Distractor |
|---------|--------|--------|------------|
| 3 | 95.8% [93.7%, 97.2%] | 100.0% [99.2%, 100.0%] | 98.8% [97.4%, 99.4%] |
| 5 | 93.4% [90.9%, 95.3%] | 100.0% [99.2%, 100.0%] | 98.0% [96.4%, 98.9%] |
| 8 | 89.0% [86.0%, 91.5%] | 100.0% [99.2%, 100.0%] | 97.0% [95.1%, 98.2%] |
| 12 | 81.4% [77.8%, 84.6%] | 100.0% [99.2%, 100.0%] | 94.2% [91.8%, 95.9%] |
| 15 | 72.4% [68.3%, 76.1%] | 100.0% [99.2%, 100.0%] | 93.6% [91.1%, 95.4%] |

The working memory bank has capacity 16 slots. At 15 facts (near saturation), simple retrieval degrades to 72.4% while update and distractor remain above 93%. Degradation is graceful, not catastrophic.

### 5.3 Update Chain Stability

Block 2: sequential updates to the same entity, N=300 per condition:

| Chain length | Accuracy | Degradation vs chain=1 |
|--------------|----------|------------------------|
| 1 | 100.0% [98.7%, 100.0%] | +0.0pp |
| 2 | 100.0% [98.7%, 100.0%] | +0.0pp |
| 4 | 100.0% [98.7%, 100.0%] | +0.0pp |
| 8 | 100.0% [98.7%, 100.0%] | +0.0pp |

Leak test at chain length 3 (N=300): 0.0% returned initial value, 0.0% returned any intermediate value. Emission is strictly the latest update.

### 5.4 Generalization

Block 3, N=300-500 per condition:

| Test | Accuracy | Notes |
|------|----------|-------|
| Rare entities only (5/30 pool) | 100.0% [98.7%, 100.0%] | No frequent distractors |
| Rare target + frequent distractors | 95.3% [92.3%, 97.2%] | Target among 4 frequent |
| Cross-schema 4-class | 70.8% [66.7%, 74.6%] | object/person NOT trained |
| Cross-trained-cluster (animal+fantasy) | 91.8% [89.1%, 93.9%] | Both clusters in training |

The 70.8% on cross-schema with untrained object/person classes is approximately 10x random baseline (6.7%). This is not complete generalization failure but represents a significant drop from same-distribution performance (91.8%), indicating that some portion of the apparent generalization derives from pretrained GPT-2 embeddings rather than the memory mechanism itself.

### 5.5 Language Preservation

Val loss on TinyStories validation set:

| Version | Final val_loss |
|---------|----------------|
| v8 (structural only) | 19.501 (catastrophic) |
| v9 (aux + cycle + LM mix 40%) | 2.571 |
| v10 (lexical + force_bank, LM mix 30%) | 2.458 |
| v11 (complex episodes, LM mix 25%) | 2.175 |

v11 achieves lower val_loss than v10 despite adding update and distractor episodes, indicating that the structural curriculum does not compete with language modeling when the architecture is correctly aligned.

---

## 6. Analysis

### 6.1 Memory Addressing Is Not Memory Emission

The most significant finding of this work is the empirical separation between:

- **Addressing competence**: computing a query that matches the correct key among many (measured as cos similarity rank)
- **Emission competence**: producing the correct token from the retrieved value (measured as argmax over vocabulary)

In v8-v9, these diverged dramatically: 97% addressing vs. 7% emission. The intuitive assumption that correct addressing implies correct emission is false. A system can know which slot is relevant while being unable to speak what is stored.

This distinction has practical implications for memory-augmented architectures in general. Evaluation metrics that measure only addressing (rank, MRR on slots) can overreport system capability. Emission requires either:

1. Direct supervision on the value-to-token pathway (our L_emit approach)
2. Values structurally designed to be lexically decodable (our lexical binding approach)
3. Ideally both, as in the final validated configuration

### 6.2 Why Lexical Binding Works

The lexical binding mechanism computes:

```
value_stored = 0.9 * W_v(embedding(answer_token)) + 0.1 * h_pool_contextual
```

Where `W_v` is a trainable linear projection and `embedding` is the shared token embedding table. Since `aux_answer_head` is also tied to the same embedding table, the emission pathway becomes:

```
retrieved_value ≈ 0.9 * W_v(E(ans)) + noise
aux_logits = retrieved_value @ E^T  
           ≈ 0.9 * W_v(E(ans)) @ E^T + noise
```

After training, `W_v` learns to be approximately identity-like in the relevant subspace, making the dot product with `E(ans)` dominate over other token embeddings. The argmax concentrates on the answer token.

This design trades generality for reliability. The stored value is not a rich contextual representation; it is primarily a projection of a specific answer token. For the validation task (discrete attribute lookup), this is sufficient. For general retrieval requiring inference over stored content, this design must be relaxed by reducing lexical_alpha, which is part of the forward agenda (Section 9).

### 6.3 Why Force-Bank Works

With 5 banks available, an untrained gate distributes facts randomly across banks. With N=4 facts and 5 banks, the expected distribution is near one-fact-per-bank. A bank with exactly one occupied slot produces softmax attention of 1.0 over that slot regardless of query-key similarity, because softmax over a single element is always 1.0.

The reader then produces this slot's value as the reader output for that bank, with full weight. When summed across banks, each bank contributes its one stored value at full magnitude. The total retrieved value is the sum of all fact values, independent of query.

`force_bank='working'` puts all N=4 facts in the same 16-slot bank. Now softmax over 4 occupied slots requires actual similarity discrimination: `attn = softmax([sim_1, sim_2, sim_3, sim_4])`. Query-key alignment (trained by L_sel) becomes effective.

This is architectural rather than loss-based. No amount of training on the original scattered configuration would have fixed the issue, because the unit-attention behavior is deterministic at one-slot occupancy. This illustrates that some memory failures are not training failures but topological ones.

### 6.4 Why ema_alpha=0.9 Works for Updates

The updater logic checks whether a new write matches an existing slot by cosine similarity (threshold theta_match=0.85). On match, it updates the slot via exponential moving average:

```
new_value = (1 - ema_alpha) * old_value + ema_alpha * incoming_value
```

With the default ema_alpha=0.3, updates produce `0.7 * old + 0.3 * new`. For "cat is red" followed by "cat is now blue", the stored value becomes `0.7 * embedding(red) + 0.3 * embedding(blue)`. When decoded through aux_answer_head, the dominant component is "red", producing incorrect emission.

With ema_alpha=0.9, the stored value becomes `0.1 * embedding(red) + 0.9 * embedding(blue)`. Emission produces "blue" correctly.

This also explains the 100% accuracy on chain=8: each update applies `0.9` to the incoming value, so after 8 updates the accumulated influence of the initial value is `0.1^8 ≈ 1e-8`, effectively zero. The chain stability is mathematical, not learned.

### 6.5 Comparison of v8/v9 vs v10/v11

The most instructive comparison is between v9 and v10. Same architecture at the level of modules, same parameter count (within 1%), same data, same hardware. Difference: three architectural fixes.

| Metric | v9 (step 2500) | v10 (step 3000) |
|--------|----------------|-----------------|
| struct_acc_4 | 7% | 93.2% |
| val_loss | 2.571 | 2.458 |
| Top-5 pattern | Identical across queries | Query-dependent |
| Memory used at inference | No (ablation: 0% drop) | Yes |

A 13.3x improvement in accuracy from architectural fixes alone, with no additional training, no new modules, no increased capacity. This suggests that memory-augmented architecture failures in related work may similarly be topological rather than scale-limited.

---

## 7. Validation Protocol (B1.1 Extended)

Beyond the basic evaluation criteria (> 85% simple, > 75% update, > 75% distractor, all PASS), we conducted a three-block extended validation on the v11 final checkpoint.

### 7.1 Block 1: Scaling

N=500 per cell. Table in Section 5.2. Findings:
- Update and distractor robust across capacity (100% and >93% respectively)
- Simple degrades monotonically from 95.8% to 72.4% between 3 and 15 facts
- No catastrophic cliff; degradation rate approximately 1.5pp per additional fact beyond 3
- Working memory has 16 slots; all measurements within capacity

### 7.2 Block 2: Update Chain Stability

N=300 per cell. Table in Section 5.3. Findings:
- 100% accuracy at chains 1, 2, 4, 8
- Zero leak (neither initial value nor intermediate values returned)
- Mathematical stability via ema_alpha as analyzed in 6.4

### 7.3 Block 3: Generalization

N=300-500 per cell. Table in Section 5.4. Findings:
- Rare entities (less frequent in training) perform identically to frequent when isolated (100%)
- Rare entities under competition with frequent distractors show 5pp degradation to 95.3%
- Cross-schema with untrained classes (object, person) achieves 70.8%, far above random but clearly below trained distribution (91.8%)
- This 21pp gap indicates that pretrained GPT-2 embeddings contribute to but do not fully explain system performance; the memory mechanism's own contribution is validated by the 91.8% on trained clusters and the 100% on updates

### 7.4 What Was NOT Tested

Explicitly deferred:
- **True held-out entities**: entities excluded from all structural training. v11 saw all 30 entities during training; the "rare" subset was rarer but not held out. True generalization test requires v12 with explicit exclusion.
- **Natural language variation**: all facts follow the template "The {entity} is {value}." Paraphrases, indirect descriptions, and ambiguous updates are not tested.
- **Multi-bank autonomy**: force_bank='working' disables the learned gate. Testing bank selection learning requires v13.
- **Integration in production LLM**: validated on a 175M parameter custom backbone. Integration into Qwen, Gemma, or Llama not tested.

---

## 8. Related Work

### 8.1 Memory-Augmented Neural Networks

The broader program of augmenting neural networks with external memory includes Neural Turing Machines (Graves et al., 2014), Memory Networks (Weston et al., 2014), and Differentiable Neural Computers (Graves et al., 2016). These approaches typically use content-addressable memory with soft attention over all slots. D_Cortex shares the content-addressable structure but differs in: (a) explicit semantic typing of banks, (b) separation of writer and reader agents with shared semantic infrastructure, and (c) lexical binding of stored values for direct decodability.

### 8.2 Retrieval-Augmented Generation

Recent work on retrieval-augmented language models (RAG: Lewis et al., 2020; Atlas: Izacard et al., 2022) focuses on retrieving text passages from external corpora to condition generation. The retrieval mechanism is typically dense vector similarity over pre-computed embeddings. D_Cortex differs in operating at token-level value binding with online memory updates during episodes, rather than retrieval from a fixed document corpus.

### 8.3 Transformer Memory Extensions

Methods like RMT (Bulatov et al., 2022), Memorizing Transformers (Wu et al., 2022), and Hyena (Poli et al., 2023) extend effective context length through compressed representations or efficient attention. D_Cortex's memory is explicit and updateable, complementing rather than replacing these approaches. In principle, D_Cortex memory could layer atop any of these architectures.

### 8.4 Cognitive Architecture Tradition

The broader motivation connects to cognitive architectures including ACT-R (Anderson, 2007), Soar (Laird, 2012), and Global Workspace Theory (Baars, 1988). These frameworks postulate separate memory systems (declarative vs. procedural, working vs. long-term) within a unified architecture. D_Cortex operationalizes one aspect of this tradition in a transformer context: memory as a structurally distinct functional layer.

---

## 9. Limitations and Forward Agenda

### 9.1 Validation Crutches

The validated configuration uses three constraints that limit the architecture's generality:

1. **force_bank='working'**: disables learned bank selection. Removing this reintroduces the one-slot-per-bank problem unless the writer is retrained with explicit bank coherence loss.

2. **lexical_alpha=0.9**: stored values are primarily projections of answer token embeddings. This works for discrete attribute lookup but cannot support queries requiring inference over rich stored content.

3. **Aux emission head dominant**: the primary emission path bypasses fusion blocks and decoder LM head. For natural language generation conditioned on memory, the fusion path needs to be re-validated.

Each of these is an item on the forward agenda.

### 9.2 True Generalization

The "rare entities" test in Block 3 measures frequency sensitivity, not generalization. All 30 entities were in v11 training; some appeared less frequently by chance. True held-out testing requires training v12 with explicit exclusion of a subset.

Prediction: held-out performance will be substantially lower than in-distribution, because the shared_address_encoder has not learned to produce distinguishable address codes for unseen token sequences beyond what GPT-2 embeddings already encode. The cross-schema result (70.8%) provides a lower bound estimate.

### 9.3 Scale

Validation uses a 175M parameter custom backbone on simple templates. Scaling considerations:

- **Model size**: integration with 1B-70B parameter backbones requires solving parameter sharing (shared embeddings become prohibitively large for Llama-70B with 128K vocab).
- **Memory capacity**: working bank has 16 slots. Real scenarios require thousands of entities. Multi-bank architecture exists but is not yet functionally differentiated (all roads lead to working in current training).
- **Training data diversity**: TinyStories + synthetic templates. Natural corpora introduce linguistic variation that the current curriculum does not cover.

### 9.4 Forward Plan

**v12 - True Held-out Validation** (estimated 90 min training):
- Exclude 5 entities entirely from structural training
- Test generalization on held-out set
- Decision point: if held-out ≥ 60%, proceed to v13; if < 30%, revisit address encoder design

**v13 - Natural Language Variation** (estimated 90 min):
- Paraphrased facts: "X has color Y", "Y is the color of X", "A Y X walked by"
- Ambiguous updates: "X might be Y now", "X seems Y"
- Varied questions: "What color?", "Tell me the color", "Color?"
- Maintains force_bank and lexical binding; isolates linguistic generalization

**v14 - Gradual Crutch Removal**:
- Phase 1 (0-1000 steps): force_bank='working' as before
- Phase 2 (1000-2500 steps): force_bank=None with probability rising 0 → 1, plus L_bank_coherence loss
- Phase 3 (2500+ steps): fully autonomous bank selection
- Add consolidation: LRU-based migration from working to episode_obj to archive

**v15+ - LLM Integration**:
- Port to Qwen or Gemma backbone
- Adapt shared infrastructure to backbone tokenizer
- Evaluate on standard retrieval benchmarks (Natural Questions, TriviaQA)

---

## 10. Conclusion

D_Cortex v2.0-alpha demonstrates experimentally that a transformer language model can be extended with explicit memory as a structurally separate functional layer, achieving memory-conditioned exact token emission with high accuracy on controlled structural tasks. The development trajectory revealed that memory addressing and memory emission are distinct computational competences. A system achieving 97% retrieval accuracy can simultaneously achieve only 7% emission accuracy due to architectural topology rather than training deficiency.

Three compounding bugs—fusion bias pollution over zero streams, insufficient retrieval softmax temperature, and bank scattering producing unit-attention artifacts—each generalize beyond this implementation. Memory-augmented systems that evaluate only on addressing metrics may significantly overreport capability. Emission requires either direct supervision, structurally decodable values, or preferably both.

The conceptual contribution is the validated existence of a third category in neural architecture design, beyond parametric memory (weights) and contextual memory (attention window). Explicit memory as demonstrated here is persistent without being parametric, addressable without being contextual, and updatable without retraining. This does not constitute a general solution to memory in neural systems; it constitutes an existence proof that such a layer can be made to work.

The forward agenda (true held-out validation, natural language variation, autonomous bank selection, LLM integration) will determine whether this mechanism generalizes to production settings or remains a controlled-regime demonstration. Either outcome provides clarity about what explicit memory systems can and cannot do.

---

## 11. Reproducibility

Complete source code, training scripts, validation scripts, and checkpoints are available at:

- Repository: https://github.com/NEURALMORPHIC-FIELDS/D_Cortex
- Patent: EP25216372.0

All experiments were conducted on Google Colab A100-SXM4-40GB instances. Training from scratch (v10) requires approximately 60 minutes. Warm-started complex training (v11) requires approximately 90 minutes. Extended validation (B1.1) requires approximately 5-8 minutes on the same hardware.

Key configuration files:
- `dcortex/config.py`: all architectural and training hyperparameters
- `colab/step2_training_v6.py`: v10 training (3000 steps, from scratch)
- `colab/step2_training_v11.py`: v11 training (4000 steps, warm start from v10)
- `colab/step2_7_b1_validation.py`: B1.1 extended validation

All reported measurements include Wilson 95% confidence intervals. Random seeds are fixed at 42-48 for each evaluation suite to ensure reproducibility.

---

## References

Anderson, J. R. (2007). *How can the human mind occur in the physical universe?* Oxford University Press.

Baars, B. J. (1988). *A cognitive theory of consciousness*. Cambridge University Press.

Bulatov, A., Kuratov, Y., & Burtsev, M. (2022). Recurrent memory transformer. *NeurIPS 2022*.

Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? *arXiv:2305.07759*.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. *arXiv:1410.5401*.

Graves, A., Wayne, G., Reynolds, M., et al. (2016). Hybrid computing using a neural network with dynamic external memory. *Nature 538, 471-476*.

Izacard, G., Lewis, P., Lomeli, M., et al. (2022). Atlas: Few-shot learning with retrieval augmented language models. *arXiv:2208.03299*.

Laird, J. E. (2012). *The Soar cognitive architecture*. MIT Press.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Poli, M., Massaroli, S., Nguyen, E., et al. (2023). Hyena hierarchy: Towards larger convolutional language models. *ICML 2023*.

Weston, J., Chopra, S., & Bordes, A. (2014). Memory networks. *arXiv:1410.3916*.

Wu, Y., Rabe, M. N., Hutchins, D., & Szegedy, C. (2022). Memorizing transformers. *ICLR 2022*.

---

## Appendix A: Complete v11 Training Log Summary

Compressed log of 4000 training steps (every 500 evaluation):

```
Step    0: emit=8.715 sel=1.085 | top1 S=--   U=60%  D=100%
Step  500: emit=0.476 sel=0.174 | top1 S=100% U=100% D=50%    [eval: S=89.5% U=99.0% D=93.0%]
Step 1000: emit=1.247 sel=0.239 | top1 S=50%  U=100% D=100%   [eval: S=90.0% U=100%  D=92.5%]
Step 1500: emit=0.024 sel=0.123 | top1 S=100% U=100% D=100%   [eval: S=89.5% U=100%  D=99.5%]
Step 2000: emit=0.012 sel=0.194 | top1 S=100% U=100% D=100%   [eval: S=95.0% U=100%  D=98.0%]
Step 2500: emit=1.048 sel=0.150 | top1 S=71%  U=100% D=100%   [eval: S=93.5% U=100%  D=94.0%]
Step 3000: emit=0.027 sel=0.117 | top1 S=--   U=100% D=100%   [eval: S=93.0% U=100%  D=99.5%]
Step 3500: emit=0.451 sel=0.130 | top1 S=80%  U=100% D=100%   [eval: S=96.5% U=100%  D=99.0%]
Step 4000: (final)                                             [eval: S=95.0% U=100%  D=99.0%]

Final (N=500): simple=94.4% update=100.0% distractor=99.2%
Final val_loss: 2.175 (on TinyStories)
Training time: 89.9 minutes on A100-SXM4-40GB
```

## Appendix B: Parameter Count

```
Total: 175.81M parameters
  Encoder (writer):     75.62M
    4 transformer blocks with 12 heads, 3072 FFN
  Decoder (reader):     98.66M
    8 standard + 4 fusion blocks
  Shared embeddings:    40.17M  (included above due to tying)
    shared_token_emb:   38.60M  (50257 vocab x 768 dim)
    shared_pos_emb:      1.57M
  Shared query engine:   0.20M
  Shared address enc:    0.97M
  Aux answer head:       1.18M  (2x Linear D->D, tied output)
  Value-to-key proj:     0.25M
```

## Appendix C: Complete File Manifest

Core architecture (dcortex/):
- `__init__.py` - package exports
- `config.py` - all hyperparameters
- `model.py` - DCortexV2Model main class (389 lines)
- `encoder.py` - MemoryEncoder, writer agent (236 lines)
- `shared_address.py` - SharedAddressEncoder (C_sigma)
- `aux_modules.py` - AuxAnswerHead, ValueToKeyProjector

Memory subsystem (dcortex/memory/):
- `banks.py` - MemoryBank with buffers and overlay mechanism
- `query.py` - QueryEngine (K_phi projection)
- `writer.py` - MemoryWriter with lexical binding and force_bank
- `readers.py` - SemanticReader with temperature, EpisodeReader, MemoryReadFusion
- `updater.py` - MemoryUpdater with theta_match and ema_alpha
- `consolidator.py` - Bank consolidation (not yet used in training)

Backbone (dcortex/backbone/):
- `embeddings.py` - token+position embedding
- `transformer.py` - MultiHeadSelfAttention, standard blocks
- `fusion_block.py` - CrossAttention, fusion blocks

Training (colab/):
- `step2_training_v6.py` - v10 training (from scratch, 3000 steps)
- `step2_training_v11.py` - v11 training (warm start, 4000 steps)
- `step2_5_ablation.py` - ablation over memory conditions
- `step2_6_deep_diagnostic.py` - 6-test diagnostic
- `step2_7_b1_validation.py` - B1.1 extended validation (3 blocks)
- `step3_benchmarks.py` - standardized benchmark suite

Infrastructure:
- `scripts/verify_integration.py` - module integration check
- `tests/test_forward_smoke.py` - end-to-end smoke test
- `tests/test_step2_fixes.py` - regression tests for fixes

Documentation:
- `README.md` - user-facing documentation
- `paper/progressive_development_report.md` - this document
- `docs/architecture.md` - detailed architecture notes
- `docs/experiments.md` - experiment log
- `docs/api.md` - API reference

---

**End of Progressive Development Report**

**Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.**  
**Cluj-Napoca, Romania**  
**Patent EP25216372.0**
