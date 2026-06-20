# D_Cortex v2.0-alpha — v15.7a SEALED

**Dual-agent memory-native transformer with longitudinal consolidation**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Patent](https://img.shields.io/badge/Patent-EP25216372.0-blue.svg)](#patent)
[![Status: v15.7a SEALED](https://img.shields.io/badge/Status-v15.7a%20SEALED-brightgreen.svg)](#status)
[![Gates](https://img.shields.io/badge/Gates-10%2F10%20green-brightgreen.svg)](paper/D_CORTEX_PAS7A_SEAL.md)

> **Memory can exist as a functional layer separate from weights and separate from context — and it can operate on its own history.**

D_Cortex demonstrates experimentally that a language model can be extended with explicit memory as a structurally separate functional layer that is persistent, addressable by content, updatable without retraining, and **capable of self-revising at episode boundaries** (consolidator pipeline: reconcile → prune → retrograde → promote).

The current sealed milestone is **v15.7a (Pas 7a, 2026-04-26)** — first longitudinal organ validated, all 10 acceptance gates green over 100 cross-episode sequences. The earlier v11 (2026-04-18) sealed memory-conditioned token emission and is preserved as the foundational layer underneath.

## Current state — the mechanism arc is PROVEN (2026-06-20)

> Full current-state summary: **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** (single source of truth).

Beyond the sealed symbolic organ, the **integration-spine** campaign (commit chain `1879c60 → 3a3c2f4`) carried the vision — *memory as the organ of thought* — end to end on the NEURAL model, measured and validity-gated:

- **Operate over persisted memory, PROVEN.** The model reaches conclusions by operating over what it persisted, with the source text ABSENT: single-step (comparison, bank-grounded) and relational **multi-hop graph traversal** (Stage 5d — a relational pointer stored as a structural COPY of the target's content-key; chaining 1.0, chain-grounded 0.949).
- **Honest at multi-hop.** `wrong_commit=0` — the signature property — extends from single facts to graph traversal: the model ABSTAINS on broken chains (Stage 5e dual gate: abstain 0.858, no-over-abstain 0.022).
- **The root was multi-object separability** — trainable into the base and generalizing over entities (Step 2, 0.92); it unlocked binding and chaining together.
- **The open frontier is free-text extraction**, MEASURED to be a **pretraining (paraphrase-robustness) property**, not a fine-tuning-diversity property, on the small from-scratch closed-vocab substrate (Stage 6–8). So extraction and scale MERGE into one move: **port the proven mechanism arc to a pretrained base.**

Per-stage falsifiable verdicts (negatives lead): `docs/STAGE_U_*`, `docs/STAGE_I_RESULT.md`, `docs/MULTIOBJECT_*`, `docs/STAGE5*`, `docs/STAGE6/7/8_*`.

## v15.x Progression — Memory Operates on Its Own History

| Pas | Date | Headline | Verdict |
|---|---|---|---|
| v15.6 Pas 3 | 2026-04 | EntitySpanComposer for F2 multiword entities | F2 0.782 |
| v15.6 Pas 6 | 2026-04-22 | RoleOfModifierResolver — entity_modifier vs attribute_value | F2 **0.952** (PASS, 7/7 gates) |
| **v15.7a** | **2026-04-26** | **Consolidator at end_episode — reconcile, prune, retrograde, promote** | **PAS 7A SEALED** (10/10 gates) |

Pas 7a defines the dynamic that distinguishes a passive store from a working memory:
- **Stable can fall**: a committed value is demoted when a challenger accumulates `M=2` distinct confirming episodes.
- **Provisional can rise**: a non-committed value is promoted when it accumulates `N=2` distinct confirmations and `K_age=2` episodes have passed.
- **Pas 6 critical path is byte-identical**: the consolidator runs only at `end_episode`, after the Pas 2/6 finalize, and never contaminates single-episode behavior.

D9 full evaluation (n=20 per L-family × 5 families = 100 sequences, A100):

```
Gate 0  trusted regression byte-identical              PASS
Gate 1  wrong_commit ≤ 0.02 across F1-F5               PASS (0.000)
Gate 2  F2 safe_resolution ≥ 0.95                       PASS (0.952)
Gate 3  false_promote_rate = 0                          PASS (0/100)
Gate 4  false_retrograde_rate = 0                       PASS (0/100)
Gate 5  L1 promote_rate ≥ 0.95                          PASS (1.000)
Gate 6  L2 retrograde_rate ≥ 0.90                       PASS (1.000)
Gate 7  L3 false_retrograde = 0 on completions          PASS (0/20)
Gate 8  L4 promote_count = 0 (anti-inflation)           PASS (0/20)
Gate 9  L5 prune_count ≥ 1 per stale trial              PASS (2/trial)

OVERALL: PAS 7A SEALED
```

Full seal certificate: [paper/D_CORTEX_PAS7A_SEAL.md](paper/D_CORTEX_PAS7A_SEAL.md).
Sealed development log: [docs/PROGRESS.md](docs/PROGRESS.md).
Self-contained Colab notebook to reproduce D9: [colab/d9_full_eval.ipynb](colab/d9_full_eval.ipynb).
Per-step sealed code: [steps/13_v15_7a_consolidation/](steps/13_v15_7a_consolidation/).

## Foundational Results (v9 → v11)

The v15.x progression builds on the v11 substrate. The original validation results below remain canonical for the foundational layer:

| Metric | v9 (broken) | v10 (principle) | v11 (complex) |
|--------|-------------|-----------------|---------------|
| Simple retrieval (4 facts) | 7% | 93.2% | **94.4%** |
| Update episodes | n/a | n/a | **100.0%** |
| Distractor (same cluster) | n/a | n/a | **99.2%** |
| Update chain (length 8) | n/a | n/a | **100.0%** |
| Val loss (TinyStories) | 2.571 | 2.458 | **2.175** |

Same architecture, same parameter count (175.81M), same data. Only three architectural fixes.

## Architecture Overview

```
  FACT TOKENS                          QUESTION TOKENS
      |                                      |
      v                                      v
  [SHARED TOKEN + POSITION EMBEDDINGS]
      |                                      |
      v                                      v
  [SHARED ADDRESS ENCODER (C_sigma)]
      |                                      |
      v                                      v
  Encoder blocks                      Decoder blocks
      |                                      |
      v                                      v
  [SHARED QUERY ENGINE (K_phi)]
      |          |
    keys       queries
      |          |
      v          v
  [MEMORY BANKS: state, episode_obj, conflict, archive, working]
                       |
                       v
                  retrieved_value
                       |
                       v
                  [AUX ANSWER HEAD]
                       |
                       v
                   ANSWER TOKEN
```

**Critical decisions:**
- Writer and reader share embeddings and query engine - structural address compatibility at init
- Stored values have explicit lexical binding: `value = 0.9 * W_v(E(answer)) + 0.1 * context`
- Retrieval softmax uses temperature 20 - sharp attention on cosine similarities
- All writes go to one bank during validation - avoids unit-attention artifact
- Emission through auxiliary head directly from retrieved value - bypasses fusion biases

## Quick Start

### Installation

```bash
git clone https://github.com/NEURALMORPHIC-FIELDS/D_Cortex.git
cd D_Cortex
pip install -r requirements.txt
```

### Running Experiments

**v10 training (validation of principle, from scratch):**
```bash
# In Colab with A100, takes approximately 60 minutes
python colab/step2_training_v6.py
```

**v11 training (complex episodes, warm start from v10):**
```bash
# Requires ckpt_step003000.pt from v10. 90 minutes.
python colab/step2_training_v11.py
```

**B1.1 extended validation on a checkpoint:**
```bash
# 5-8 minutes, generates full B1.1 report
python colab/step2_7_b1_validation.py
```

### Using the Model

```python
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
import torch

cfg = DCortexConfig()
model = DCortexV2Model(cfg)

ckpt = torch.load('checkpoints/ckpt_v11_step004000.pt', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

model.reset_memory()
model.begin_episode()

fact_tokens = tokenize("The cat is red.")
answer_token_id = tokenize(" red")[0]

model.encode(
    fact_tokens,
    answer_token_id=answer_token_id,
    lexical_alpha=0.9,
    force_bank='working',
)

question_tokens = tokenize("What color is the cat? The cat is")

logits, retrieved_value = model.decode(
    question_tokens,
    return_retrieved=True,
)

aux_logits = model.aux_answer_head(retrieved_value)
predicted_token = aux_logits.argmax(dim=-1).item()
```

## Directory Structure

```
D_Cortex/
├── README.md                     # This file
├── MISIUNEA.txt                  # Mission statement (foundational, do not modify)
├── LICENSE                       # Proprietary license
├── CITATION.cff                  # Citation information
├── CHANGELOG.md                  # Version history
├── QUICKSTART.md                 # Quick run instructions
├── requirements.txt              # Python dependencies
│
├── dcortex/                      # Core library (v11 base substrate)
│   ├── config.py                 # DCortexConfig: all hyperparameters
│   ├── model.py                  # DCortexV2Model main class
│   ├── encoder.py                # Writer agent (memory encoder)
│   ├── shared_address.py         # SharedAddressEncoder (C_sigma)
│   ├── aux_modules.py            # AuxAnswerHead, ValueToKeyProjector
│   ├── memory/                   # Memory subsystem
│   │   ├── banks.py              # MemoryBank with buffers + overlay
│   │   ├── query.py              # QueryEngine (K_phi)
│   │   ├── writer.py             # MemoryWriter with lexical binding
│   │   ├── readers.py            # SemanticReader + Fusion
│   │   ├── updater.py            # Theta-match + EMA updates
│   │   └── consolidator.py       # Bank consolidation (latent layer)
│   └── backbone/                 # Standard transformer components
│
├── steps/                        # Sealed development steps (v15.x progression)
│   ├── 08-10_v15_5_to_v15_6_pas3/    # Historical bundle: v15.5 holdout + Pas 3
│   ├── 12_v15_6_pas6_romr/            # Pas 6 SEALED: RoMR, F2 0.952
│   ├── 13_v15_7a_consolidation/       # Pas 7a SEALED: consolidator, 10/10 gates
│   │   ├── README.md             # Step-level spec + status
│   │   ├── SEAL.md               # Seal certificate (citable artifact)
│   │   ├── NOTES.md              # Internal dev journal
│   │   └── code.py               # Sealed source (18k+ lines, full pipeline)
│   └── _template/                # Template for future steps
│
├── colab/                        # Colab notebooks + training scripts
│   ├── d9_full_eval.ipynb        # Self-contained Pas 7a D9 reproducer (1MB)
│   ├── step2_training_v6.py      # v10 training (from scratch)
│   ├── step2_training_v11.py     # v11 training (warm start)
│   ├── step2_5_ablation.py       # Ablation on memory conditions
│   ├── step2_6_deep_diagnostic.py # 6-test diagnostic
│   ├── step2_7_b1_validation.py  # B1.1 extended validation
│   └── step3_benchmarks.py       # Standard benchmark suite
│
├── docs/
│   ├── PROGRESS.md               # Sealed-step development log (read this for history)
│   ├── architecture.md           # Detailed architecture documentation
│   └── experiments.md            # Experiment log
│
├── paper/
│   ├── progressive_development_report.md  # Full scientific report (v9-v11)
│   ├── technical_note_three_bugs.md       # Standalone note: 3 architectural bugs
│   └── D_CORTEX_PAS7A_SEAL.md             # Pas 7a seal certificate
│
├── api.md                        # API reference
├── architecture.md               # Architecture (root copy, mirrors docs/)
├── experiments.md                # Experiment log (root copy)
├── progressive_development_report.md      # Scientific report (root copy)
├── technical_note_three_bugs.md           # Three bugs note (root copy)
├── raport_stiintific_dcortex_dezvoltare_progresiva.pdf  # PDF version (RO)
│
├── scripts/
│   └── verify_integration.py     # Module integration check
└── tests/
    ├── test_forward_smoke.py     # End-to-end smoke test
    └── test_step2_fixes.py       # Regression tests
```

**Reading order for new arrivals:**
1. [README.md](README.md) (this file) — overview
2. [docs/PROGRESS.md](docs/PROGRESS.md) — chronological development log
3. [paper/D_CORTEX_PAS7A_SEAL.md](paper/D_CORTEX_PAS7A_SEAL.md) — current seal certificate
4. [steps/13_v15_7a_consolidation/README.md](steps/13_v15_7a_consolidation/README.md) — Pas 7a spec
5. [paper/progressive_development_report.md](paper/progressive_development_report.md) — foundational v9-v11 report

## Development Timeline

**April 16, 2026**: Initial v1-v5 exploration. Dual-agent architecture established.

**April 17, 2026 (morning)**: v6 with separate encoder/decoder and overlay mechanism for gradient flow. v7 with shared semantic infrastructure. Both failed.

**April 17, 2026 (afternoon)**: v8 introduces structural curriculum. Retrieval metrics reach 97% but emission stays at 7%.

**April 17, 2026 (evening)**: v9 adds auxiliary answer head, cycle loss, LM mix. Same plateau. Deep diagnostic reveals three root causes.

**April 17-18, 2026 (night)**: v10 fixes three architectural bugs. Validation of principle: 93.2% top-1 on 4-fact retrieval.

**April 18, 2026 (morning)**: v11 adds complex episodes. All criteria pass with margin.

**April 18, 2026 (afternoon)**: B1.1 extended validation with Wilson 95% CI.

See [paper/progressive_development_report.md](paper/progressive_development_report.md) for complete scientific report.

## What Was Demonstrated

1. **Memory-conditioned exact token emission** (not just distributional influence)
2. **Memory addressing vs emission are distinct competences** (97% vs 7% is possible)
3. **Three generalizable architectural bugs** that cause emission failure:
   - Fusion projection bias pollution over zero streams
   - Insufficient retrieval softmax temperature
   - Bank scattering produces unit-attention artifact
4. **Update chain stability** via ema_alpha=0.9 (mathematical, not learned)
5. **Compositional limit** with cross-schema untrained classes (70.8% vs 91.8%)

## What Was NOT Demonstrated

1. **True held-out generalization** - all entities seen in training
2. **Autonomous bank selection** - force_bank='working' used as validation crutch
3. **Natural language variation** - fixed "The X is Y" template
4. **LLM integration** - 175M custom backbone only
5. **Multi-bank functional differentiation** - state/episode/conflict/archive dormant

See "Forward Agenda" in [paper/progressive_development_report.md](paper/progressive_development_report.md) Section 9.

## Status

**Current sealed version**: **v15.7a (Pas 7a, April 26, 2026)** — first longitudinal organ.
**Foundational checkpoint**: `ckpt_v11_step004000.pt` (April 18, 2026, B1.1 validated).
**Validation status**: 10/10 D9 acceptance gates green. Pas 6 trusted regression byte-identical under Pas 7a arbiter.
**Next milestone**: undecided — either Pas 7b (semantic abstraction layer that produces hypotheses for the consolidator to metabolize) or Pas 8 (integration of D_Cortex 7a as longitudinal backend for an explicit organism). Neither begins until an explicit adapter is defined.

## Reproducibility

All experiments run on NVIDIA A100-SXM4-40GB (Google Colab):
- v10 training: approximately 60 minutes from scratch
- v11 training: approximately 90 minutes with warm start
- B1.1 validation: approximately 5-8 minutes

Random seeds fixed at 42-48 across evaluation suites. Wilson 95% CI reported on all accuracy measurements. Checkpoints are deterministic given seeds and hardware.

Data: TinyStories (Eldan & Li, 2023) - 80M train tokens, 4.8M val tokens via Hugging Face.

## Patent

This architecture and its components are covered under European Patent EP25216372.0 (FHRSS - Fractal Holographic Redundant Storage System), held by FRAGMERGENT TECHNOLOGY S.R.L. The memory substrate, lexical value binding, and shared address encoder constitute claim elements. All uses require licensing agreement with FRAGMERGENT TECHNOLOGY S.R.L.

## License

Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L. All rights reserved.

See [LICENSE](LICENSE) for terms.

## Citation

```bibtex
@techreport{borbeleac2026dcortex,
  title = {D_Cortex v2.0-alpha (v15.7a SEALED): A Dual-Agent Memory-Native
           Transformer with Longitudinal Consolidation},
  author = {Borbeleac, Vasile Lucian},
  institution = {FRAGMERGENT TECHNOLOGY S.R.L.},
  year = {2026},
  month = {April},
  address = {Cluj-Napoca, Romania},
  version = {v15.7a},
  url = {https://github.com/NEURALMORPHIC-FIELDS/D_Cortex},
  note = {Patent EP25216372.0. Pas 7a sealed 2026-04-26, all 10 D9
          acceptance gates green over 100 cross-episode sequences.
          See paper/D_CORTEX_PAS7A_SEAL.md.},
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation.

## Contact

**Vasile Lucian Borbeleac**  
Founder, FRAGMERGENT TECHNOLOGY S.R.L.  
Cluj-Napoca, Romania

For research collaboration, licensing, or technical questions, please open an issue on GitHub or contact through FRAGMERGENT TECHNOLOGY S.R.L. directly.

## Acknowledgments

This work builds on the foundational FRAGMERGENT framework (fragment + emergence + convergence) developed by Vasile Lucian Borbeleac. The architecture incorporates insights from the FHRSS patent (EP25216372.0) regarding fault-tolerant storage geometry, and extends concepts from the "Dinamica Ființei" philosophical work on object formation and persistent identity.

---

*D_Cortex v2.0-alpha (v15.7a SEALED): memory as a third category in neural architecture — and one that operates on its own history.*
