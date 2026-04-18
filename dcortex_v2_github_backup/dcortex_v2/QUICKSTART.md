# Quickstart Guide

Fast-track guide to running D_Cortex v2.0-alpha experiments.

## Prerequisites

- Google Colab account (Pro recommended for A100 access) OR local machine with A100 40GB+
- GitHub account
- Google Drive for persistent storage

## Step 1: Clone Repository

```bash
git clone https://github.com/FRAGMERGENT/dcortex-v2.git
cd dcortex-v2
```

## Step 2: Setup Colab Environment

Upload any of the `colab/step*.py` files to Colab. The scripts are self-contained and include all dependencies, source file generation, and data loading.

**Mount Google Drive** (first cell in each script does this automatically):
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Project root is hardcoded to**: `/content/drive/MyDrive/dcortex_v2`

The script creates these subdirectories automatically:
- `checkpoints/` - model checkpoints
- `logs/` - training logs
- `dataset_cache/` - cached tokenized data
- `results/` - evaluation outputs
- `training_data/` - local data if any
- `scripts/` - utility scripts
- `configs/` - configuration presets

## Step 3: Run Your First Experiment

### Option A: Quick Validation (5-8 minutes)

If you already have a trained checkpoint in Google Drive:

```bash
# Run B1.1 extended validation
python colab/step2_7_b1_validation.py
```

This validates the model on:
- Scaling (3/5/8/12/15 facts)
- Update chains (length 1/2/4/8)
- Generalization (rare entities, cross-schema, cross-trained-cluster)

Output: JSON report in `/content/drive/MyDrive/dcortex_v2/results/b1_1_validation.json`

### Option B: Train v10 From Scratch (60 minutes)

```bash
# Fresh training, validation of principle
python colab/step2_training_v6.py
```

Expected outcome:
- 3000 training steps
- Final: top1 = 93.2% on 4-fact retrieval (N=500)
- Checkpoint saved: `ckpt_step003000.pt`

### Option C: Complex Episodes (90 minutes, requires v10 checkpoint)

```bash
# Warm-start from v10
python colab/step2_training_v11.py
```

Expected outcome:
- 4000 training steps
- Final: simple=94.4%, update=100%, distractor=99.2%
- Checkpoint saved: `ckpt_v11_step004000.pt`

## Step 4: Understand the Results

### Training Output Format

```
Step   500/4000 | emit=0.705 sel=0.364 lm=0.522 | top1 S=100% U=100% D=50% | n=8/3/2/3 | gn=4.92 ETA 78m
```

- `emit`: L_emit loss (cross-entropy on emission)
- `sel`: L_sel loss (selection supervision)
- `lm`: LM loss (when in LM mode)
- `top1 S=/U=/D=`: running top-1 accuracy by episode type (Simple, Update, Distractor)
- `n=`: count of each type in last 16 steps (S/U/D/LM)
- `gn`: gradient norm
- `ETA`: estimated time remaining

### Evaluation Output Format

```
[EVAL] step=2000 | val=2.498 | simple=95.0% update=100.0% distr=98.0%
```

- `val`: validation loss on TinyStories held-out
- `simple/update/distr`: top-1 accuracy on 200 evaluation episodes each type

## Step 5: Debug Failed Runs

If you see:

### Error: "No v11 checkpoint found"

```bash
# Check Drive sync (Colab local cache may be stale)
ls -la /content/drive/MyDrive/dcortex_v2/checkpoints/
```

If empty but you see files in Drive UI, force remount:
```python
drive.flush_and_unmount()
drive.mount('/content/drive')
```

### Error: OOM (Out of Memory)

This shouldn't happen on A100 40GB with 175M param model. If it does:
- Reduce `accumulation_steps` from 16 to 8
- Reduce `seq_len` from 64 to 32

### top1 stuck at 7% for many steps

This is the v9 failure mode. Three architectural bugs present. Check:
- `force_bank='working'` is passed to `model.encode()`
- Reader temperature τ=20 in `SemanticReader.forward()`
- `retrieved_value = r_state + r_episode + r_conflict + r_archive + r_working` (direct sum)
- `lexical_alpha=0.9` passed to encode

See `paper/technical_note_three_bugs.md` for detailed diagnosis.

## Step 6: Integrate Into Your Work

### Using trained model in your application

```python
import torch
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model

cfg = DCortexConfig()
model = DCortexV2Model(cfg).cuda().eval()

# Load checkpoint
ckpt = torch.load('ckpt_v11_step004000.pt', weights_only=False)
model.load_state_dict(ckpt['model'])

# Process episode
def process_episode(facts, question):
    model.reset_memory()
    model.begin_episode()
    
    for fact_text, answer_tok_id in facts:
        fact_tokens = tokenize(fact_text)
        ans_id = torch.tensor([answer_tok_id]).cuda()
        model.encode(
            fact_tokens.cuda(),
            answer_token_id=ans_id,
            lexical_alpha=0.9,
            force_bank='working',
        )
    
    q_tokens = tokenize(question).cuda()
    _, retrieved = model.decode(q_tokens, return_retrieved=True)
    logits = model.aux_answer_head(retrieved)
    return logits.argmax(dim=-1).item()
```

### Running on TPU

Not tested. The architecture uses standard PyTorch primitives but `torch.compile` is not compatible with all memory operations. TPU evaluation would likely need adaptation.

### Running with Float32

```python
# In training script, change this:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    ...

# To this:
# (remove autocast)
```

Caveat: ~2x slower, ~2x more memory.

## Common Modifications

### Change Episode Format

Episodes are generated in training scripts via `generate_simple_episode()`, `generate_update_episode()`, `generate_distractor_episode()`. Modify these functions to test different task structures. Keep the same interface: return `(facts, prompt, target_answer_text, target_answer_token_id)`.

### Change Entity/Attribute Pool

Edit `_ENTITIES` and `_COLORS` lists at the top of training scripts. Pool sizes affect difficulty. Minimum viable: 5 entities, 5 values.

### Add New Bank

1. Add capacity in `config.py`: `new_bank_capacity: int = 32`
2. Add to `DCortexV2Model.__init__`: `self.new_bank = MemoryBank(...)`
3. Add to `BANK_ORDER` in `MemoryWriter`
4. Add reader in `DCortexV2Model.decode`
5. Add to `MemoryReadFusion`
6. Include in `retrieved_value` sum

## Reading Scientific Output

After B1.1 validation, the generated report contains:

```json
{
  "block1_scaling": {
    "n_3": {
      "simple": {"acc": 0.958, "ci": [0.937, 0.972], "n": 500},
      "update": {"acc": 1.000, "ci": [0.992, 1.000], "n": 500},
      "distractor": {"acc": 0.988, "ci": [0.974, 0.994], "n": 500}
    },
    ...
  },
  "block2_updates": { ... },
  "block3_generalization": { ... }
}
```

Wilson 95% confidence intervals: Use lower bound for conservative reporting, upper bound for best-case estimation.

## Next Steps

- Read `paper/progressive_development_report.md` for full scientific context
- Read `docs/architecture.md` for architectural details
- Read `docs/experiments.md` for development history
- Read `paper/technical_note_three_bugs.md` for the bug investigation

---

**Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.**  
**Patent EP25216372.0**
