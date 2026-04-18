# Changelog

All notable changes to D_Cortex v2.0-alpha are documented in this file.

Format: keep a changelog style. Dates in ISO 8601. Semantic versioning loosely applied.

## [v11] - 2026-04-18

### B1.1 Extended Validation Complete

Extended validation with Wilson 95% confidence intervals across three blocks:

- Block 1 (Scaling): simple 95.8% -> 72.4% at n_facts 3 -> 15, update 100% throughout, distractor 98.8% -> 93.6%
- Block 2 (Update Chains): 100% at chain length 1, 2, 4, 8. Zero leak.
- Block 3 (Generalization): Rare-only 100%, rare+distractors 95.3%, cross-schema 4-class 70.8%, cross-trained-cluster 91.8%

### Added
- `colab/step2_7_b1_validation.py`: comprehensive three-block validation script
- `paper/progressive_development_report.md`: full scientific report
- `paper/technical_note_three_bugs.md`: standalone technical note on architectural bugs
- `docs/architecture.md`: detailed architecture documentation
- `docs/experiments.md`: experiment log
- `docs/api.md`: API reference

### Complex Episodes Training
- 4000 steps, 89.9 minutes on A100-SXM4-40GB
- Warm start from `ckpt_step003000.pt` (v10)
- Final metrics (N=500): simple=94.4%, update=100%, distractor=99.2%
- val_loss improved from 2.458 (v10) to 2.175 (v11)

### Changed
- `ema_alpha`: 0.3 -> 0.9 (updates replace rather than blend)
- Training episode mix: 50% simple / 25% update / 25% distractor, LM ratio 25%
- LR multipliers reduced (shared 5x -> 3x, encoder 3x -> 2x) for warm start

## [v10] - 2026-04-17

### Validation of Principle

First version to achieve non-trivial memory-conditioned emission. Three architectural bugs identified and fixed:

### Fixed
1. **Fusion projection bias pollution**: `retrieved_value = sum(raw_reader_outputs)` instead of passing through `nn.Linear` projections with bias
2. **Reader softmax temperature**: hardcoded `tau=20` in `SemanticReader.forward()` to sharpen attention over cosine similarities in [-1, 1]
3. **Bank scattering**: added `force_bank='working'` parameter to writer - all structural writes to single bank to avoid unit-attention artifact from one-slot-per-bank occupancy

### Changed
- `query_weights`: `(0.5, 0.3, 0.2)` -> `(1.0, 0.0, 0.0)` - matches L_sel supervision
- Added lexical value binding: `value = 0.9 * W_v(E(answer)) + 0.1 * context`
- Aux answer head tied to shared_token_emb for direct emission

### Results
- 3000 steps training, 60.6 minutes on A100
- struct_acc_4: 93.2% (N=500), from 7% in v9
- Scaling: 3f=95.7%, 4f=94.7%, 5f=90.3%, 6f=90.3%, 8f=85.3%, 10f=78.7%
- val_loss: 2.458

### Added
- `colab/step2_training_v6.py`: v10 training script
- `colab/step2_5_ablation.py`: ablation over memory conditions
- `colab/step2_6_deep_diagnostic.py`: 6-test diagnostic suite

## [v9] - 2026-04-17

### Failed Plateau

Added auxiliary head + cycle loss + LM mix. Retrieval remained at 97%, emission at 7%. Top-5 tokens identical across different queries, indicating memory not functionally used at inference.

### Added
- `AuxAnswerHead` module with tied projection
- `ValueToKeyProjector` for L_cycle loss
- Curriculum ratio 0.6 (60% structural, 40% LM)

## [v8] - 2026-04-17

### Retrieval-Emission Gap

Pure structural curriculum achieves 97% retrieval accuracy (key-query alignment) but 7% token emission. First empirical confirmation that memory addressing and memory emission are distinct competences.

### Added
- Structural training curriculum with L_sel, L_sep, L_occ losses
- Memory condition ablation testing

## [v6-v7] - 2026-04-17

### Foundation

Shared semantic infrastructure (shared embeddings, shared address encoder, shared query engine). Overlay mechanism for gradient flow through memory.

## [v1-v5] - 2026-04-16

### Initial Exploration

Dual-agent architecture prototypes. Various failure modes explored.

---

## Forward Plan

### [v12] - Planned

**True Held-Out Validation**:
- Exclude 5 entities entirely from structural training
- Test generalization on held-out set
- Decision point: if held-out >= 60%, proceed to v13; if < 30%, revisit address encoder

### [v13] - Planned

**Natural Language Variation**:
- Paraphrased facts: "X has color Y", "Y is the color of X"
- Ambiguous updates: "X might be Y now"
- Varied questions: "What color?", "Tell me the color"
- Tests linguistic generalization independently of architectural autonomy

### [v14] - Planned

**Gradual Crutch Removal**:
- Phase-out force_bank='working' across training
- Add L_bank_coherence loss
- Implement consolidation: working -> episode_obj -> archive
- Tests autonomous bank selection

### [v15+] - Planned

**LLM Integration**:
- Port to Qwen, Gemma, or Llama backbone
- NQ/TriviaQA benchmark evaluation

---

**Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.**
