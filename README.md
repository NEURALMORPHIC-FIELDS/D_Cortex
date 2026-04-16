# D_Cortex v2.0-alpha

**Memory-native transformer architecture.**

D_Cortex is a from-scratch transformer in which long-term memory is
*structurally integrated into the model*, not appended to a frozen
backbone through hooks. This repository contains the Step 1
"executable architecture" milestone: every module is defined,
connected, and produces real output in an end-to-end forward pass.

Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
Patent EP25216372.0. Cluj-Napoca, Romania.

---

## 1. Why v2.0

DECORTEX v1.x (v1.3b / v1.4 / v1.5 / v1.6) bolted a memory system onto a
frozen Qwen backbone via a single-layer hook. It worked as a proof of
concept but hit four structural ceilings that no amount of patching
could remove:

1. **Lossy compression.** Entire conversations collapsed into ~16 state
   slots plus one SSM vector plus a handful of conflict slots.
2. **Single-layer hook fusion.** Memory was injected at one backbone
   layer, not integrated iteratively through the stack.
3. **Static phase A / phase B.** The memory was written in phase A and
   read in phase B; no live updates during a conversation.
4. **No semantic addressing.** Memory was cross-attended as a flat
   block, not queried with structured keys.

v2.0-alpha answers each of these at the architectural level. It is a
rewrite, not a patch.

| v1.x limit                    | v2.0-alpha response                                              |
| ----------------------------- | ---------------------------------------------------------------- |
| Lossy compression             | Six explicit banks with distinct semantics and capacities.       |
| Single-layer hook fusion      | `FusionBlock` as a *native* backbone layer in the last N layers. |
| Static phase A / phase B      | Live writer + updater on every forward pass.                     |
| No semantic addressing        | `QueryEngine` produces `(q_ent, q_rel, q_typ)`; readers use NN-semantic attention. |

---

## 2. Architecture

### 2.1 Memory (six components)

```
M_state         slot-based stable facts                  capacity 64
M_episode_obj   discrete episodic objects                capacity 128
x_t_ep          EpisodeSSM: continuous recurrent state   state_dim 256
M_conflict      difference vectors for contradictions    capacity 32
M_archive       long-term consolidated storage           capacity 512
M_work          rolling short-term working memory        capacity 16
```

Each slot-based bank stores a latent key triplet `(k_ent, k_rel, k_typ)`
and a value vector. Keys and values are PyTorch buffers updated
in-place by the writer. Gradients enter the memory subsystem through
the softmax-weighted read
`softmax(q @ k.T) @ v`, where `q` comes from the trainable
`QueryEngine`.

`EpisodeSSM` is a trainable state-space recurrence:

```
x_t = sigmoid(a) * x_{t-1} + B * phi(u_t)
r_t = C * x_t
```

with learned `a`, `B`, `C` and `phi = GELU`. The state `x` is
persistent across forward passes but detached between turns, so no
cross-turn graph accumulates.

### 2.2 Read path (five streams + sub-fusion)

```
             +------------- q_ent, q_rel, q_typ (QueryEngine)
             |
  M_state ---+---> SemanticReader ---> r_state
  M_ep_obj --+
  x_t_ep ----+---> EpisodeReader (W_theta sub-fusion) ---> r_episode
  M_conflict-+---> SemanticReader ---> r_conflict
  M_archive -+---> SemanticReader ---> r_archive
  M_work ----+---> SemanticReader ---> r_working

  r_state, r_episode, r_conflict, r_archive, r_working
         |
         v
    MemoryReadFusion    --->  memory_tokens [B, 5, D]
```

`EpisodeReader` internally fuses object-read and SSM-readout through a
dedicated `W_theta` submodule before the global `MemoryReadFusion`
stacks all five streams as a five-token "memory context" for
cross-attention inside each `FusionBlock`.

### 2.3 Write path

```
  h_pool (pooled final hidden state)
        |
        v
  MemoryWriter
        |
        +--- gate (softmax over 6 options)
        +--- value_head (MLP)
        +--- key_ent, key_rel, key_typ  (three Linear heads)
        |
        v
  argmax(gate) --> {state, episode_obj, conflict, archive, working, skip}
        |
        v
  MemoryUpdater.update(bank, value, keys, step)
        |
        +--- if bank == state: also run detect_conflict -> dual write
        +--- if bank == conflict: store (value - existing) as diff vector
        |
        v
  Allocation policy:
      1. free slot exists and similarity < theta_match -> allocate free
      2. similarity >= theta_match -> EMA update (or diff write)
      3. bank full, no match -> evict LRU
```

### 2.4 Backbone

```
input_ids
    |
  TokenEmbeddings (token + abs learned positional, pre-norm, dropout)
    |
  StandardTransformerBlock  x (n_layers - n_fusion_layers)
    |
  pool -> QueryEngine -> (q_ent, q_rel, q_typ)
    |
  five memory reads + MemoryReadFusion -> memory_tokens [B, 5, D]
    |
  FusionBlock               x n_fusion_layers
    |     (self-attn + cross-attn-to-memory + FFN, gated by mem_gate)
    |
  pool -> MemoryWriter -> updater mutates banks (torch.no_grad)
    |
  LayerNorm + LM head (weights tied to token embedding)
    |
  logits [B, T, vocab_size]
```

Default scale:

| Parameter          | Value |
| ------------------ | ----- |
| `vocab_size`       | 50257 |
| `hidden_dim`       | 768   |
| `n_layers`         | 12    |
| `n_heads`          | 12    |
| `ff_dim`           | 3072  |
| `max_seq_len`      | 2048  |
| `n_fusion_layers`  | 4     |

See `dcortex/config.py` for the full `DCortexConfig` and
`DCortexConfig().small_test()` for the CI-sized tiny config.

---

## 3. Repository layout

```
dcortex_v2/
├── .claude/                          project memory (append-only)
│   ├── project_concept.json          vision, distinction from v1.x, principles
│   ├── project_structure.json        live module blueprint
│   └── project_log.json              chronological decision log
├── dcortex/                          library package
│   ├── __init__.py
│   ├── config.py                     DCortexConfig dataclass
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── banks.py                  six memory banks + EpisodeSSM
│   │   ├── query.py                  QueryEngine
│   │   ├── updater.py                MemoryUpdater (NN-semantic match)
│   │   ├── readers.py                SemanticReader, EpisodeReader, MemoryReadFusion
│   │   ├── writer.py                 MemoryWriter (6-way gate)
│   │   └── consolidator.py           decay + prune/migrate + pairwise merge
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── embeddings.py             token + positional embeddings
│   │   ├── transformer.py            MHSA, FFN, StandardTransformerBlock
│   │   └── fusion_block.py           CrossAttention, FusionBlock
│   └── model.py                      DCortexV2Model wrapper
├── scripts/
│   └── verify_integration.py         mandatory integration verifier
├── tests/
│   └── test_forward_smoke.py         end-to-end smoke test
└── README.md                         this file
```

---

## 4. Requirements

* Python 3.10 or newer
* PyTorch 2.0 or newer (tested with 2.11)

No other hard dependencies. The library is deliberately torch-only.

Install torch (if not already present):

```bash
pip install torch
```

---

## 5. Quick start

### 5.1 Verify the codebase

Every commit should pass `verify_integration.py` before any progress
claim. It parses `dcortex/model.py`, walks the import graph, and
classifies each `.py` file as `WIRED` / `NOT_WIRED` / `DUPLICATE`.

```bash
python scripts/verify_integration.py
```

Expected: `exit 0`, all critical modules `WIRED`.

### 5.2 Run the smoke test

```bash
python tests/test_forward_smoke.py
```

This instantiates the model with the tiny `small_test` config, runs a
forward pass, verifies the output shape, runs five forwards to
populate memory, resets memory, runs consolidation, and performs a
backward pass. All stages print their own `✓` line.

### 5.3 Use the model

```python
import torch
from dcortex import DCortexConfig, DCortexV2Model

# Tiny config for experimentation; use DCortexConfig() for the full 12L/768d model.
cfg = DCortexConfig().small_test()
model = DCortexV2Model(cfg)

input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(input_ids)               # [2, 16, vocab_size]

# Memory snapshot at any point:
print(model.memory_snapshot())

# Session control:
model.reset_memory()                     # clear all banks and SSM state
_ = model.consolidate()                  # run one consolidation pass
```

To disable writes for a particular forward (e.g. at evaluation time):

```python
logits = model(input_ids, write_memory=False)
```

---

## 6. Memory semantics at a glance

* **Key triplet.** Every write produces three latent keys
  `(k_ent, k_rel, k_typ)` from the pooled hidden state. Similarity at
  read and update time is a weighted cosine combination in these three
  spaces, with default weights `(0.5, 0.3, 0.2)`.
* **Thresholds.** `theta_match = 0.7` gates update vs allocate.
  `theta_conflict = 0.3` is the value-cosine cutoff that triggers
  conflict routing (same key signature, divergent value).
* **Allocation policy.** Free slot first, then EMA update on match,
  then LRU eviction. Diff vectors in `ConflictMemory` are not blended.
* **Gradients.** Keys and values are buffers, so no gradients flow
  into memory storage itself. Gradients flow into `QueryEngine` (read
  side) and into the writer's gate / value / key heads through
  downstream losses.
* **Consolidation.** `MemoryConsolidator` decays `usage`, prunes
  slots below `consolidate_prune_threshold` (optionally migrating to
  archive), and greedily merges pairs above
  `consolidate_merge_threshold`. No learnable parameters.

---

## 7. Development rules (abridged)

These are enforced by file headers and by `verify_integration.py`:

* Code, comments, variables, docstrings, filenames, and docs are
  English only.
* Every standalone `.py` file carries the copyright header (omitted
  from `__init__.py` and test files).
* Absolute imports only.
* Type annotations on every function signature.
* No `logging` module. `print()` with status prefixes
  (`✓`, `[INFO]`, `[WARN]`, `[ERROR]`, `[HF]`).
* No em dashes in text. Commas, colons, or parentheses instead.
* File integrity: never create empty or placeholder files.
* JSON files under `.claude/` are append-only. Prior entries are
  never summarized, compressed, or deleted.

---

## 8. Roadmap

### Step 1: Executable architecture (complete)

Every module is defined with a functional `forward()`. The full flow
runs end-to-end on random input and produces logits of the correct
shape. Acceptance:

* `verify_integration.py` exits 0 with all critical modules `WIRED`.
* `tests/test_forward_smoke.py` passes, including backward.

### Step 2: Training loop (planned)

Full training loop on the native architecture (not a reduced
variant). Scope:

* Data pipeline with local-SSD staging (nanoGPT pattern) for the
  backbone, plus a synthetic curriculum that stress-tests the memory
  subsystem (long-horizon recall, conflict resolution, continual
  update).
* Loss composition: LM cross-entropy plus auxiliary memory losses
  (write, abstain, sparsity on the writer gate, conflict recall).
* Optimizer, LR schedule with warmup and decay, gradient clipping,
  gradient accumulation.
* Checkpointing with atomic writes, numeric-step sort, and exact
  resume.
* Mixed precision: bfloat16 on A100-class GPUs, fp16 + GradScaler on
  T4/L4.

### Step 3: Benchmarks, scaling, curriculum (planned)

* Memory-targeted benchmarks (long-horizon fact recall, conflict
  resolution, continual learning), contrasted with Qwen-frozen
  baselines and the v1.x lineage.
* Synthetic curriculum generator.
* Scale study at 12L/768d and beyond.

---

## 9. Role of Qwen

Qwen is **not** part of the D_Cortex v2.0-alpha forward pass. It is
retained only as:

* an external baseline for benchmarks,
* a candidate teacher for synthetic data in Step 2, and
* a comparator in ablation studies.

No Qwen weights, hooks, or modules are imported by the `dcortex`
package.

---

## 10. Citation and IP

* Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
* Patent EP25216372.0 (FHRSS and related architectural components).
* Contact: FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania.

All rights reserved. Redistribution and use in source and binary forms
require written permission from the copyright holder.
