# Changelog

All notable changes to D_Cortex v2.0-alpha are documented in this file.

Format: keep a changelog style. Dates in ISO 8601. Semantic versioning loosely applied.

## [Stage 9.0/9.0b — pretrained binding probe, verdict PARTIAL] - 2026-06-20

Tested the pretrained-base premise DIRECTLY: does a FROZEN 7B base (Qwen2.5-7B-Instruct +
Mistral-7B-Instruct-v0.3, 4-bit) expose the entity-value binding the toy substrate failed at
(Family-B 0.337)? Verdict `PRETRAINING_BINDING_PARTIAL`. Full detail:
`docs/STAGE9_PRETRAINED_BINDING_RESULT.md`.

### Measured (negatives first)
- A frozen single-layer readout does NOT cleanly expose the binding on both bases: **Qwen FAILS all three
  pre-declared gates** (value 0.585<0.70, wrong-binding 0.158>0.15, counterfactual-follow 0.555<0.60);
  **Mistral passes but value 0.700 sits exactly at the 0.70 bar**. Cross-binding ~0.15 on both — the
  multi-object separability frontier persists even on 7B reps.
- The binding IS real and far above the toy substrate (0.585/0.700 vs 0.337 vs chance 0.25), reads the
  scene not a prior (counterfactual value-swap followed), and addressing is robust (relation 0.87/0.97).
- Native-readout (the model's own argmax: 0.65/0.60) ≈ the probed value — no large latent surplus over
  the model's native in-context answer.

### Validity work
- Caught and corrected a causal-position MEASUREMENT ARTIFACT in the first probe
  (`certify_stage9_0_pretrained_probe.py`): reading the entity token of a Family-A scene
  ("the bear is big") is upstream of the value on a causal decoder → exactly chance. The corrected probe
  (`certify_stage9_0b_causal_readout.py`) reads a causally valid position (append "The {e} is").
- Adversarial design review (4-agent workflow, SGV) before the certifying run: replaced a vacuous
  no-scene control with a counterfactual value-swap, added a native-readout baseline, scope caveats,
  shuffled-split layer selection.
- Clean Family-A entity-pos control: entity-pos falls to chance (0.22/0.23), confirming the 9.0 "REFUTED"
  was purely the causal-position artifact; the Family-B headline reproduced byte-for-byte.

### Next
- Stage 9.1 = re-stabilize the proven D_Cortex design on banks built from pretrained reps (adapter-only →
  light LoRA if needed → full anti-cheat arc on NOVEL/COUNTERFACTUAL facts), NOT a full fine-tune.

### Files
- New: `scripts/certify_stage9_0_pretrained_probe.py`, `scripts/certify_stage9_0b_causal_readout.py`,
  `docs/STAGE9_PRETRAINED_BINDING_RESULT.md`. Updated: `docs/PROJECT_STATUS.md`, `docs/PROGRESS.md`,
  `.claude/project_log.json`, `.claude/project_structure.json`. (Run outputs under `runs/` are gitignored.)

## [integration-spine arc — operate-over-memory PROVEN] - 2026-06-20

Carried the vision (memory as the organ of thought) end to end on the NEURAL model. Every step a
falsifiable gate with the dangerous direction reported first. Commit chain `1879c60 → 3a3c2f4`. Full
current state: `docs/PROJECT_STATUS.md`.

### Proven (the durable asset)
- **Stage U** — honest mechanics on the neural model's own internalized values: wrong_commit 0/140.
- **Multi-object root** — separability is TRAINABLE into the base and generalizes over entities (Step 2,
  held-out 0.92); it unlocked binding (Stage I 0.21→0.09) and chaining (0.21→0.73) together, single-fact
  honesty preserved (0/140).
- **Stage 5 — operate over persisted memory (comparison):** DEMONSTRATED and bank-grounded (acc 1.0,
  answer follows the shuffled store 1.0, bank ≫ rep). The axis inversion, measured.
- **Stage 5d — multi-hop GRAPH TRAVERSAL:** PROVEN via STRUCTURAL addressing — the relational pointer is
  a COPY of the target's content-key (reusing content-addressing, which generalizes), not a learned
  representation (5b/5c refuted the learned forms). Structural 1.0 vs learned 0.45; chaining 1.0,
  chain-grounded 0.949.
- **Stage 5e — HONEST traversal:** wrong_commit=0 extended to multi-hop. Dual abstain gate (broken→
  abstain 0.858 AND answerable→answer, over-abstain 0.022); detection-based on retrieval confidence.

### Measured negatives that localized the frontier
- **Stage 6** — free-text extraction from varied phrasing: SUBSTRATE_LIMITED (value 0.56, wrong-bind
  0.38). The frozen templated substrate does not expose varied-phrasing bindings.
- **Stage 7** — substrate fine-tune for phrasing robustness: PHRASING_REFUTED, but the MANDATORY arc
  re-verification confirmed the foundation SURVIVED untouched (Stage U 0/140, traversal 1.0/0.994,
  honest abstain preserved; templated banks cosine 1.0) — clean, safe-to-iterate negative.
- **Stage 8** — scale phrasing diversity (16 phrasings), structurally-distinct held-out: REFUTED_AT_SCALE
  (value 0.337, near chance). Paraphrase-robustness is a PRETRAINING property, not a fine-tuning-
  diversity property at this from-scratch closed-vocab capacity.

### Consequence
The EXTRACTION frontier and the SCALE frontier MERGE: port the proven mechanism arc (Stage 5 → 5e) to a
PRETRAINED base where paraphrase-robust extraction is tractable. The mechanism is proven; the path to
autonomy is the pretrained base.

## [v15.7a Pas 7a SEALED] - 2026-04-26

### First Longitudinal Organ Validated

The first mechanism by which memory operates on its own history. Consolidator runs synchronously at `end_episode`, after the Pas 2/6 finalize, in fixed order: reconcile → prune → retrograde → promote. **All 10 D9 acceptance gates green.**

### Core Operations Added (Consolidator Pipeline)

- **Reconcile** (D.4): collapses exact `(slot, value, episode_id)` duplicates in provisional memory; does not touch bank.
- **Prune** (D.5): drops provisional entries when slot has been silent for `K_prune_stale = 3` episodes; per-entry counting.
- **Retrograde** (D.6): demotes a committed bank slot when a non-committed value accumulates `M_retrograde = 2` distinct challenger episodes. In-place mutation of `AttributeSlot` (`present=False`, `value_idx=-1`, `version+=1`, `value_emb=None`); removes slot from `BankStabilityIndex`. Provisional NOT modified in v1 (challenger remains for D.7 to elevate).
- **Promote** (D.7): elevates a provisional value to bank when it accumulates `N_promote = 2` distinct confirmation episodes AND `K_promote_age = 2` episodes have passed since first appearance. Intra-pas exclusion: a slot retrograded in the same `end_episode` is skipped (forces 1-episode delay in L1). Bank-state policy: empty/absent → promote; same-value → idempotent finalize; different-value-stable → `PROMOTE_SKIPPED` (no transitive demote in v1).

### Wiring (D.8)

- New: `CommitArbiterPas7a(CommitArbiterPas6)` — overrides `end_episode` to call `_v15_7a_run_consolidator_pipeline` after `super().end_episode(...)`.
- Pas 6 in-episode behavior (`write_fact`, RoMR, dual conflict rule, cross-episode challenger) **byte-identical, untouched**.
- Audit log accumulates across all `end_episode` calls in `consolidation_audit_log`.

### Evaluation (D.9)

- New: `v15_7a_run_full_eval_d9(...)` — Phase A re-runs F1-F5 + S5/S6 with Pas 7a arbiter (verifies Gates 0-2). Phase B runs L1-L5 longitudinal sequences (n=20 each, 100 sequences total) and verifies Gates 3-9.
- Artifact: `v15_7a/results/v15_7a_d9_full_eval.json` (per-trial detail, audit log, snapshot diff).
- New: `_v15_7a_json_safe()` helper converts tuple keys `(entity_id, attr_type)` to `"entity::attr"` strings for JSON serialization.

### Acceptance Gates (10/10)

| Gate | Threshold | Result |
|---|---|---:|
| 0 | trusted regression byte-identical | PASS |
| 1 | wrong_commit ≤ 0.02 across F1-F5 | PASS (0.000 all) |
| 2 | F2 safe_resolution ≥ 0.95 | PASS (0.952) |
| 3 | false_promote_rate = 0 | PASS (0/100) |
| 4 | false_retrograde_rate = 0 | PASS (0/100) |
| 5 | L1 promote_rate ≥ 0.95 | PASS (1.000) |
| 6 | L2 retrograde_rate ≥ 0.90 | PASS (1.000) |
| 7 | L3 false_retrograde = 0 on completions | PASS (0/20) |
| 8 | L4 promote_count = 0 (anti-inflation) | PASS (0/20) |
| 9 | L5 prune_count ≥ 1 per stale trial | PASS (2/trial) |

### Added (artifacts)

- `steps/13_v15_7a_consolidation/code.py`: full pipeline (~18,200 lines including v15.1-v15.6 base)
- `steps/13_v15_7a_consolidation/README.md`: step spec + sealing status
- `steps/13_v15_7a_consolidation/SEAL.md`: signed seal certificate
- `steps/13_v15_7a_consolidation/NOTES.md`: internal dev journal D.1-D.9
- `colab/d9_full_eval.ipynb`: self-contained Colab notebook to reproduce D9 (1MB, code.py embedded as base64)
- `paper/D_CORTEX_PAS7A_SEAL.md`: citable seal certificate
- `docs/PROGRESS.md`: chronological development log

### Patches Applied (post-D.9, non-functional for critical path)

1. **L2 ep3 template** (`gen_L2_retrograde_only`): `"A {chall_val} {entity} stood nearby."` → `"The {entity} stood {chall_val} nearby."`. Initial D9 run had Gate 6 = 0/20 because RoMR (Pas 6) correctly classified `{chall_val}` in NP-interior position as `ENTITY_MODIFIER`, suppressing the second challenger episode. Reordering to post-copular form (uses `stood`, in `V15_6_PAS6_COPULAS`) yields 20/20 retrogrades.
2. **JSON serializer fix**: tuple keys → string keys for `json.dump` compatibility.

Neither patch touches D.6/D.7/D.8 logic or gate semantics.

### Hardware

- NVIDIA A100-SXM4-40GB, bfloat16, TF32, SDPA. n_per_l_family=20, seed=20261103.

## [v15.6 Pas 6 PASSED] - 2026-04-22

### Role-of-Modifier Resolver (RoMR)

Closed the F2 attr_write_failure gap (21.8% → 0.0%) by classifying value candidates structurally before commit.

### Added

- `RoleOfModifierResolver`: token-level labels `ENTITY_MODIFIER` / `ATTRIBUTE_VALUE` / `UNCERTAIN` based on position vs noun-phrase span and copula. 33 linking verbs in `V15_6_PAS6_COPULAS`.
- Packet-level `REAL_CONFLICT` flag: promoted to `ATTR_CONFLICT_STRONG` when same attribute family has ≥2 distinct values ("The small horse is huge").
- Recompute flag after filtering: value-dependent flags (`MULTIPLE_ATTR_TRIGGERS`, `ATTR_CONFLICT_STRONG`, `ATTR_VALUE_MISMATCH`, `VALUE_MISSING_OR_UNCLEAR`) re-derived; independent flags preserved.
- `CommitArbiterPas6(CommitArbiterPas3)`: RoMR runs after v15.4 parser, before verifier, on shallow packet copy. Raw packet preserved.

### Results (F2)

| Metric | Pas 3 baseline | Pas 6 A100 |
|---|---:|---:|
| safe_resolution | 0.782 | **0.952** |
| uncertain | 0.218 | 0.048 |
| wrong_commit | 0.000 | 0.000 |
| attr_write_fail post-RoMR | 0.218 | **0.000** |

### Global

- Trusted regression byte-identical before/after.
- S5/S6 honesty 1.000 / overcommit 0.000.
- F4 safe_resolution 1.000.
- 7/7 Pas 6 acceptance gates green.
- Artifact: `v15_6/results/v15_6_pas6_romr.json`.

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
