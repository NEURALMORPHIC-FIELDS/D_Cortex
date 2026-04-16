# Step 2 Training Notebook -- User Technical Review
# Date: 2026-04-16
# Source: user message in session 2a56c99f, line 388 of JSONL transcript
# Status: CRITICAL FEEDBACK, NOT YET ADDRESSED

## Verdict

Varianta lui Claude este competenta ca notebook de antrenare generala, dar nu
este suficient de fidela repo-ului tau si nu valideaza comprehensiv arhitectura
de memorie asa cum pretinde.

### Classification
- good first training notebook: YES
- faithful to real repo: NO (partially)
- comprehensive memory validation: NO
- A100-optimal / Flash Attention 2: NO (false claim)

---

## What is good

1. General training config is reasonable (bf16, no GradScaler, TF32, AdamW,
   warmup+cosine, gradient clipping, atomic checkpointing, memmap batching)
2. Uses the complete model, not a mock
3. Logging is decent (train loss, eval loss, ppl, gate entropy, occupancy,
   grad norm, throughput)

---

## 10 Problems (in order of severity)

### P1. Will crash on import with real repo [RESOLVED]
The notebook assumes `dcortex/` package structure. User initially thought
repo was flat files at root. VERIFIED: dcortex/ package structure EXISTS
correctly on disk and GitHub. This point is NOT a real issue.

### P2. Does NOT use Flash Attention 2, despite claiming it
Code in transformer.py and fusion_block.py uses manual attention:
  q @ k.transpose(...) -> softmax -> attn @ v
Notebook does NOT patch to F.scaled_dot_product_attention, does NOT install
flash-attn, does NOT rewrite modules. Only delivers bf16 + TF32.

### P3. Memory ablation uses DIFFERENT random batches
memory_ablation() compares loss with populated vs empty memory but draws
different random batches for each phase. This introduces noise. Correct:
fix same batches, run once with memory, once after reset.

### P4. Per-submodule gradient tracking MISSING
Claims "Gradients flow through all modules" but only measures global
grad_norm. Does not prove gradient flow through: query engine, readers,
writer heads, fusion blocks, embeddings, lm head individually.
Need: grad norm per submodule, count of params with nonzero grad.

### P5. No memory curriculum
Trains almost exclusively standard LM on TinyStories, then inspects memory
statistics. Missing explicitly:
- fact update tasks
- contradiction tasks
- delayed recall tasks
- overwrite vs conflict tests
- episode continuity tests

### P6. ConflictMemory semantics NOT validated
The most important architectural feature (difference-vector storage when
key matches but value diverges) is not tested, measured, or compensated.

### P7. Gate entropy pressure does not prove semantic routing
Minimizing gate entropy makes the gate more decisive, but this could be
an artifact of loss pressure, not evidence of correct semantic routing.

### P8. Gate metrics from LAST micro-batch only
After grad_accum, reads _gate_store['probs'] which corresponds to the
last forward in accumulation, not the average across all micro-batches.
Partial and slightly misleading metric.

### P9. No SDPA optimization
At seq_len=1024, model materializes full [B, H, T, T] attention scores.
Without SDPA/flash patch: more memory, lower throughput, NOT "A100 optimal".

### P10. Memory validation is activity-based, not semantic
Measures occupancy, route distribution, gate entropy. This says the system
writes somewhere. Does NOT say it writes correct information, recovers it
correctly, or that conflict memory represents useful differences.

---

## Required Corrections (user's exact words, in priority order)

1. runtime package normalization for real repo [RESOLVED - not needed]
2. patch attention to SDPA for A100 flash kernels
3. same-batch memory ablation, not random-vs-random
4. grad-flow metrics per submodule, not just global norm
5. memory curriculum real: fact insertion, delayed recall,
   contradiction/update, overwrite vs conflict routing
6. explicit test for ConflictMemory semantics
7. correct aggregation of gate metrics across all micro-batches
