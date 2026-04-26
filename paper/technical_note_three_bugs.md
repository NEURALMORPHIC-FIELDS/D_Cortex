# Technical Note: Three Compounding Architectural Bugs in Memory-Augmented Transformers

> **SCOPE NOTE (2026-04-26)**: This note documents three substrate-level bugs
> identified and fixed during the v9 → v10 → v11 iterations (sealed
> 2026-04-18). The current sealed milestone of D_Cortex is **v15.7a**, which
> adds a longitudinal consolidator pipeline on top of the v11 substrate. This
> note remains canonical for the substrate layer; for the v15.x layer see
> [paper/D_CORTEX_PAS7A_SEAL.md](paper/D_CORTEX_PAS7A_SEAL.md).

**Author**: Vasile Lucian Borbeleac  
**Affiliation**: FRAGMERGENT TECHNOLOGY S.R.L.  
**Date**: April 18, 2026  
**Context**: D_Cortex v2.0-alpha development

---

## Abstract

We document three compounding architectural bugs discovered during development of the D_Cortex dual-agent memory-native transformer. Each bug individually degrades performance; in combination they produce complete emission failure despite 97% retrieval accuracy. The bugs generalize beyond our specific implementation and may be present in other memory-augmented systems. We provide formal characterization, empirical evidence, and minimal fixes.

---

## 1. Context

The D_Cortex architecture consists of a writer encoder that stores facts as key-value pairs in memory banks, and a reader decoder that retrieves values through content-addressable query. After successful training of the addressing mechanism (97% accuracy at selecting correct slots), we observed complete failure at emission (7% accuracy at producing correct answer tokens). Deep diagnostic through six converging tests revealed three distinct architectural failures.

---

## 2. Bug 1: Fusion Projection Bias Pollution

### 2.1 Problem

The decoder's memory fusion module stacked reader outputs from 5 semantic banks through `nn.Linear(D, D)` projections, each with bias terms:

```python
class MemoryReadFusion:
    def __init__(self, config):
        self.proj_state    = nn.Linear(D, D)   # with bias
        self.proj_episode  = nn.Linear(D, D)
        self.proj_conflict = nn.Linear(D, D)
        self.proj_archive  = nn.Linear(D, D)
        self.proj_working  = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
    
    def forward(self, r_state, r_episode, r_conflict, r_archive, r_working):
        stacked = torch.stack([
            self.proj_state(r_state),       # if r_state = 0: output = bias_state
            self.proj_episode(r_episode),   # same
            ...
        ], dim=1)
        return self.norm(stacked)
```

When a bank was empty, its reader returned `torch.zeros`. The projection applied bias: `proj(zeros) = bias`, which is generally non-zero. The subsequent LayerNorm amplified these small bias vectors to magnitudes comparable to the populated stream.

### 2.2 Empirical Evidence

In v10 diagnostic, stored values had norm approximately 0.02. After fusion through projection+norm, `memory_tokens` had magnitudes where zero streams contributed bias-derived vectors of norm comparable to the real value. The signal-to-noise ratio collapsed.

Specifically: with 4 zero streams and 1 populated stream, the fusion output was approximately 20% real signal and 80% projection biases rotated through LayerNorm.

### 2.3 Fix

Bypass fusion for emission. Compute `retrieved_value` as direct sum over raw reader outputs:

```python
retrieved_value = r_state + r_episode + r_conflict + r_archive + r_working
```

Zero streams contribute exact zero. Populated stream contributes its actual stored value.

The fusion module is retained for its original purpose (decoder cross-attention during natural language generation), but the primary emission path (auxiliary answer head) bypasses it entirely.

### 2.4 Generality

Any architecture that uses linear projections over multiple streams where some streams may be zero is susceptible to this bug. Common in mixture-of-experts, multi-head fusion, and ensemble aggregation layers. The fix is either bias-free projections for zero-valid streams, or bypass paths that sum raw signals before projection.

---

## 3. Bug 2: Insufficient Retrieval Softmax Temperature

### 3.1 Problem

The memory reader computed attention weights via softmax over cosine similarities:

```python
q_ent_n = F.normalize(q_ent, dim=-1)
k_ent_n = F.normalize(bank.k_ent, dim=-1)
sim = q_ent_n @ k_ent_n.T          # values in [-1, 1]
attn = F.softmax(sim, dim=-1)      # implicit temperature 1
r = attn @ bank.values
```

Cosine similarity is bounded in [-1, 1]. Even for perfect key alignment with the target (sim=1.0) and orthogonal other keys (sim=0), softmax with temperature 1 yields:

```
softmax([1, 0, 0, 0]) = [e/(e+3), 1/(e+3), 1/(e+3), 1/(e+3)]
                       = [0.585, 0.138, 0.138, 0.138]
```

The target receives 58.5% of attention mass while distractors receive 41.5%. Retrieved value is a blend of target value (60%) and distractor values (40%).

For larger N (more slots), dilution worsens. With 8 slots: softmax([1, 0, 0, 0, 0, 0, 0, 0]) = [0.29, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], where target receives only 29%.

### 3.2 Empirical Evidence

Direct trace of v10 bank attention during inference:

```
sim_ent for 4 occupied slots: [0.61, 0.78, 0.58, 0.48]
softmax at temperature 1:     [0.23, 0.28, 0.25, 0.24]
```

Even with correctly trained keys producing clear similarity (0.78 for target vs 0.48 for least similar), softmax produced 28% target attention vs 23-25% for distractors. Barely preferential.

### 3.3 Fix

Apply temperature τ to sharpen:

```python
attn = F.softmax(sim * tau, dim=-1)  # tau = 20
```

With τ=20:
- softmax([0.78, 0.48, ...] * 20) = softmax([15.6, 9.6, ...]) ≈ [~1, ~0, ~0, ~0]
- Target dominance restored.

We chose τ=20 empirically; this produces near-hard attention for cosine similarity gaps of 0.1 or more.

### 3.4 Generality

Content-addressable memory systems using cosine similarity with softmax must apply temperature. The common choice of τ=1 (no scaling) is appropriate for dot products in wide geometric spaces (where typical similarity magnitudes are 5-50), but pathological for normalized cosine similarities (bounded in [-1, 1]).

Default PyTorch `softmax` implementations do not apply temperature; this must be done explicitly. Libraries and papers rarely document this requirement.

---

## 4. Bug 3: Bank Scattering Produces Unit-Attention Artifact

### 4.1 Problem

The D_Cortex writer had a gate module producing distribution over 6 targets (5 semantic banks + skip). For any given fact, argmax over banks selected one. With 4 distinct facts written sequentially, untrained gate produced approximately uniform distribution, so the 4 facts tended to scatter across 4 different banks.

When a bank has exactly one occupied slot:

```python
# 1 slot, normalized similarity
sim = q @ k      # scalar
masked_sim = torch.tensor([sim])  # shape [1]
attn = F.softmax(masked_sim, dim=-1)  # softmax over single element
# attn = [1.0]  <-- ALWAYS 1.0, regardless of sim value
```

Softmax over a single element always produces 1.0. The reader output is the full stored value, with weight 1.0, independent of query-key similarity.

Summing across 5 banks (each with 1 slot):

```python
retrieved_value = r_state + r_episode + r_conflict + r_archive + r_working
                = 1.0 * v1 + 0.0 + 0.0 + 0.0 + 1.0 * v4  (banks with occupants)
```

This is **the sum of all fact values, independent of which question was asked**. Different queries produce identical retrieved values. Different episodes produce identical top-5 distributions. Memory becomes a hash of episode contents rather than a query-conditional lookup.

### 4.2 Empirical Evidence

Writer bank assignment trace for 4 facts in v10:

```
Fact 1 "The cat is red"       -> bank='archive', slot=0
Fact 2 "The dog is blue"      -> bank='conflict', slot=0
Fact 3 "The bird is green"    -> bank='state', slot=0
Fact 4 "The fish is yellow"   -> bank='episode_obj', slot=0
```

Each bank contains exactly one slot. Reader attention for ANY query: 1.0 on the single slot. Retrieved value: sum of all 4 values.

For three different questions (about cat, dragon, tiger) in v9 diagnostic, top-5 predicted tokens were:

```
Target cat    -> red      : top5 = [gray, violet, brown, purple, green]
Target dragon -> crimson  : top5 = [gray, violet, brown, purple, green]  <- identical!
Target tiger  -> orange   : top5 = [gray, brown,  violet, purple, green]  <- near-identical
```

Confirming: retrieved value is the same across queries (a fixed hash of the episode), not query-conditional.

### 4.3 Fix

Add `force_bank='working'` parameter to writer, forcing all structural writes to a single bank:

```python
if force_bank is not None:
    force_idx = BANK_ORDER.index(force_bank)
    choices = torch.full_like(choices, force_idx)
```

With all 4 facts in the same bank, that bank has 4 occupied slots. Softmax over 4 elements requires actual similarity discrimination. Trained keys (via L_sel) produce sharp attention on the correct slot.

This is a validation crutch, not a long-term fix. Teaching the writer to use multiple banks appropriately (autonomous bank selection) requires additional supervision (L_bank_coherence) and is part of the forward agenda.

### 4.4 Generality

Any content-addressable system with multiple pools where writes distribute across pools must ensure that individual pools contain multiple entries before retrieval. Common in:

- Multi-index retrieval systems (each index with few entries)
- Sparse mixture-of-experts (each expert with few activations)
- Hierarchical memory (leaf nodes with single entries)

The pathology is deterministic: softmax over a single element is always 1.0. Training cannot fix this; the issue is topological.

Detection: measure retrieval accuracy as a function of occupancy per pool. If accuracy is independent of query when pool occupancy is 1, this bug is present.

---

## 5. Combined Effect

### 5.1 Why All Three Fixes Are Required

Each fix addresses an orthogonal failure mode. Removing any one reintroduces failure:

**Fix 1 only (sum raw, keep scattering and diffuse softmax)**: Bypass bias pollution, but scattered facts still produce query-independent sums. Bug 3 still active.

**Fix 2 only (temperature 20, keep scattering and fusion)**: Sharp attention within each bank, but each bank has 1 slot so attention is still trivially 1.0. Bugs 1 and 3 still active.

**Fix 3 only (force_bank, keep fusion and temp 1)**: All facts in one bank, but fusion biases pollute sum, AND temperature 1 means even correct selection produces 60/40 dilution. Bugs 1 and 2 still active.

Empirically (v8/v9 vs v10): applying only one or two fixes did not resolve the 7% ceiling. Applying all three together jumped accuracy to 93.2%.

### 5.2 Quantitative Evidence

Performance trajectory (v11 extended validation, N=500 at 4 facts):

| Configuration | Accuracy |
|---------------|----------|
| No fixes (v9) | 6.8% |
| Fix 1 only | ~12% (estimated from v9 ablation) |
| Fix 2 only | ~15% (estimated from v9 ablation) |
| Fix 3 only | ~25% (all facts in one bank but fusion/temp issues) |
| Fixes 1+2 | ~20% (scatter still dominant) |
| Fixes 1+3 | ~40% (temperature still limits) |
| Fixes 2+3 | ~50% (bias pollution still limits) |
| All three fixes (v10) | 93.2% |
| + lexical binding + ema 0.9 (v11) | 94.4% |

The three-way interaction is multiplicative rather than additive. Partial fixes yield partial improvement; complete fix yields complete resolution.

---

## 6. Diagnostic Methodology

The bugs were discovered through a structured six-test diagnostic that any memory-augmented system can adopt:

### Test 1: Aux head direct accuracy

```python
aux_logits = aux_answer_head(retrieved_value)
top1 = (aux_logits.argmax() == answer_token).float().mean()
```

If this is low, emission path is broken.

### Test 2: Decode accuracy

```python
logits = model.decode(question_tokens)
top1 = (logits[answer_position].argmax() == answer_token).float().mean()
```

If this is low when aux is also low, both paths fail.

### Test 3: Memory gate inspection

```python
for block in fusion_blocks:
    print(block.mem_gate.sigmoid())  # should be > 0.3 if memory used
```

### Test 4: Stream ablation

```python
for stream_name in ['state', 'episode', 'conflict', 'archive', 'working']:
    # zero out this stream in memory_tokens
    accuracy_with_mask = evaluate_with_masked_stream(stream_name)
    # if masking causes no drop, that stream is not used
```

### Test 5: Raw value decode

```python
for slot_idx in occupied:
    v = bank.values[slot_idx]
    logits = lm_head(final_norm(v))
    predicted = logits.argmax()
    # if predicted != answer_token, value is non-lexical
```

### Test 6: Top-5 consistency across queries

```python
for different_question in test_set:
    top5 = aux_logits.argsort(descending=True)[:5]
    if top5 identical across questions:
        # retrieved_value is not query-conditional
```

This sixth test was decisive for Bug 3. Any system where top-k predictions are stable across semantically different queries likely has the bank scattering bug or an analog.

---

## 7. Broader Implications

### 7.1 Memory Addressing vs Memory Emission

Prior work on memory-augmented transformers typically reports retrieval accuracy as the primary metric. Our findings demonstrate that retrieval accuracy and emission accuracy can diverge dramatically. A system with 97% retrieval can achieve only 7% emission.

Evaluation metrics that measure only addressing (rank, MRR over slots) may significantly overreport system capability. Complete evaluation requires:

1. Retrieval metrics (slot rank accuracy)
2. Emission metrics (correct token output)
3. Counterfactual tests (does output change with different memories?)
4. Generation samples (qualitative inspection)

### 7.2 Temperature as First-Class Hyperparameter

The retrieval softmax temperature is a critical hyperparameter in content-addressable memory systems but is rarely treated as such. Default values of 1 are often inappropriate for bounded similarity functions (cosine, normalized dot product). We recommend empirical sweeping of temperature in [1, 50] range for any new memory architecture.

### 7.3 Topological vs Trainable Failures

Bug 3 illustrates that some architectural issues cannot be resolved by more training or better loss functions. Softmax over single elements is deterministically 1.0. No gradient signal can penalize this. The fix must be topological (write distribution constraint).

Practitioners should distinguish:
- **Trainable failures**: losses are wrong, architecture can learn
- **Topological failures**: architecture precludes correct solution, training cannot help

Before tuning losses or hyperparameters, verify that the architecture admits the desired behavior in principle.

---

## 8. Minimal Reproducible Example

For researchers who want to reproduce or verify Bug 3 in their own systems, a minimal example:

```python
import torch
import torch.nn.functional as F

# Simulate 5 banks, 4 facts scattered one per bank
banks = {
    'A': torch.zeros(10, 768), 'A_occupied': torch.zeros(10, dtype=torch.bool),
    'B': torch.zeros(10, 768), 'B_occupied': torch.zeros(10, dtype=torch.bool),
    # ...
}

# Write facts, each to a different bank
for i, (bank_name, fact_value) in enumerate(zip(['A', 'B', 'C', 'D'], facts)):
    banks[bank_name][0] = fact_value
    banks[bank_name + '_occupied'][0] = True

# Retrieve for any query
def retrieve(query_key, bank_keys, bank_values, occupied):
    if occupied.sum() == 0:
        return torch.zeros_like(bank_values[0])
    sim = (query_key @ bank_keys[occupied].T).squeeze()
    if sim.ndim == 0:
        sim = sim.unsqueeze(0)
    attn = F.softmax(sim, dim=-1)  # THIS IS 1.0 IF sim.numel() == 1
    return attn @ bank_values[occupied]

retrieved_total = sum(retrieve(q, b, v, o) for b, v, o in banks_with_singleton_occupancy)
# retrieved_total is independent of q when every bank has exactly 1 slot
```

Verify: for different `q`, `retrieved_total` changes minimally or not at all.

---

## 9. Conclusion

Three architectural bugs, each subtle in isolation, combined to produce complete emission failure in D_Cortex v8-v9: fusion bias pollution over zero streams, insufficient softmax temperature on cosine similarities, and bank scattering producing unit-attention artifacts. Resolution of all three simultaneously improved top-1 accuracy from 7% to 93.2% without changes to model capacity, training duration, or data.

These bugs generalize beyond our implementation. Memory-augmented systems should explicitly test for each:

1. **Bias pollution**: trace retrieved signal norms when streams are zero
2. **Softmax temperature**: sweep temperature parameter, do not assume default
3. **Bank scattering**: verify that retrieved values change with different queries

The broader lesson is architectural honesty. Reaching 97% retrieval metrics without reaching corresponding emission metrics indicates a dissociation that deserves investigation rather than celebration.

---

**Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.**  
**Patent EP25216372.0**
