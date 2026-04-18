# D_Cortex v2.0-alpha

**Dual-agent memory-native transformer architecture**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Patent](https://img.shields.io/badge/Patent-EP25216372.0-blue.svg)](#patent)
[![Status](https://img.shields.io/badge/Status-v11%20validated-green.svg)](#status)

> **Memory can exist as a functional layer separate from weights and separate from context.**

D_Cortex v2.0-alpha demonstrates experimentally that a language model can be extended with explicit memory as a structurally separate functional layer that is persistent, addressable by content, and updatable without retraining.

## Key Results

After 11 development iterations and resolution of three compounding architectural bugs:

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
git clone https://github.com/FRAGMERGENT/dcortex-v2.git
cd dcortex-v2
pip install torch tiktoken datasets matplotlib numpy
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
dcortex-v2/
├── dcortex/                      # Core library
│   ├── __init__.py
│   ├── config.py                 # All hyperparameters
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
│   │   └── consolidator.py       # Bank consolidation (not yet active)
│   └── backbone/                 # Standard transformer components
│       ├── embeddings.py
│       ├── transformer.py
│       └── fusion_block.py
├── colab/                        # Training and evaluation scripts
│   ├── step2_training_v6.py      # v10 training (from scratch)
│   ├── step2_training_v11.py     # v11 training (warm start)
│   ├── step2_5_ablation.py       # Ablation on memory conditions
│   ├── step2_6_deep_diagnostic.py # 6-test diagnostic
│   ├── step2_7_b1_validation.py  # B1.1 extended validation
│   └── step3_benchmarks.py       # Standard benchmark suite
├── scripts/
│   └── verify_integration.py     # Module integration check
├── tests/
│   ├── test_forward_smoke.py     # End-to-end smoke test
│   └── test_step2_fixes.py       # Regression tests
├── paper/
│   └── progressive_development_report.md  # Full scientific report
├── docs/
│   ├── architecture.md           # Detailed architecture documentation
│   └── experiments.md            # Experiment log
├── configs/                      # Configuration presets
├── results_archive/              # Historical results JSONs
├── README.md                     # This file
├── LICENSE                       # Proprietary license
├── CITATION.cff                  # Citation information
├── .gitignore
└── requirements.txt
```

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

**Current version**: v11 (April 18, 2026)  
**Current checkpoint**: `ckpt_v11_step004000.pt`  
**Validation status**: B1 PASS with margin, B1.1 Extended Validation complete  
**Next milestone**: v12 True Held-out Training

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
  title = {D_Cortex v2.0-alpha: A Dual-Agent Memory-Native Transformer Architecture},
  author = {Borbeleac, Vasile Lucian},
  institution = {FRAGMERGENT TECHNOLOGY S.R.L.},
  year = {2026},
  month = {April},
  address = {Cluj-Napoca, Romania},
  note = {Patent EP25216372.0. Progressive Development Report.},
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

*D_Cortex v2.0-alpha: memory as a third category in neural architecture.*
