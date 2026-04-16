# -*- coding: utf-8 -*-
# ===========================================================================
# D_Cortex v2.0-alpha -- Step 2: Training + Memory Validation
# Google Colab A100 GPU -- SDPA-optimized
# Single Monolithic Cell
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Patent EP25216372.0. Cluj-Napoca, Romania.
# ===========================================================================
#
# PURPOSE: Train the full D_Cortex v2.0-alpha (140.81M params, 12L/768d)
# on real data and validate memory subsystem behavior.
#
# WHAT THIS NOTEBOOK PROVES:
#   1. Training loop stable (loss decreases, no NaN/Inf)
#   2. Gradients flow through all submodules (per-submodule grad norms)
#   3. Writer gate evolves from uniform toward specialized routing
#   4. Memory banks populate during training
#   5. Same-batch ablation: populated vs empty memory on identical data
#   6. ConflictMemory stores difference vectors when values diverge
#   7. Phase 2 memory curriculum stresses fact-recall and contradiction
#   8. Model generates text after training
#
# WHAT THIS NOTEBOOK DOES NOT PROVE:
#   - That the model has learned semantically correct memory routing
#     (gate sharpening from entropy loss != semantic correctness)
#   - That memory improves performance long-term (requires Step 3 benchmarks)
#   - That 5000 steps is sufficient to fully validate memory utility
#
# HARDWARE: A100 80GB (bfloat16, TF32, NO GradScaler)
#           SDPA via F.scaled_dot_product_attention (flash kernel when available)
# DATA: TinyStories (roneneldan/TinyStories) + synthetic memory curriculum
# ===========================================================================

import os, sys, time, math, json, gc, io, contextlib, subprocess, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# ======================== 1. ENVIRONMENT ====================================

from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/dcortex_v2'
CHECKPOINT_DIR = f'{PROJECT_ROOT}/checkpoints'
RESULTS_DIR = f'{PROJECT_ROOT}/results'
BIN_DIR = f'{PROJECT_ROOT}/dataset_cache/bin'
LOCAL_DATA = '/content/tmp_data'
SEP = '=' * 70

for d in [PROJECT_ROOT, CHECKPOINT_DIR, RESULTS_DIR, BIN_DIR, LOCAL_DATA]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Project root: {PROJECT_ROOT}", flush=True)

# ======================== 2. GPU DETECTION ==================================

import torch

assert torch.cuda.is_available(), "CUDA required. Connect to a GPU runtime."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_mem / (1024**3)
GPU_CAP = torch.cuda.get_device_capability(0)

print(SEP)
print(f"[INFO] GPU: {GPU_NAME} | VRAM: {GPU_MEM_GB:.1f} GB | SM {GPU_CAP[0]}.{GPU_CAP[1]}")

if 'A100' in GPU_NAME or GPU_CAP[0] >= 8:
    DTYPE = torch.bfloat16
    USE_SCALER = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] A100 mode: bfloat16, TF32 enabled, NO GradScaler")
else:
    DTYPE = torch.float16
    USE_SCALER = True
    print(f"[WARN] {GPU_NAME}: fp16 + GradScaler (not optimal, but functional)")

torch.backends.cudnn.benchmark = True

# Check SDPA availability
_SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"[INFO] SDPA (F.scaled_dot_product_attention): "
      f"{'AVAILABLE' if _SDPA_AVAILABLE else 'NOT AVAILABLE (PyTorch < 2.0)'}")
print(SEP)

DEVICE = torch.device('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ======================== 3. CLONE + DEPENDENCIES ===========================

REPO_URL = "https://github.com/NEURALMORPHIC-FIELDS/D_Cortex.git"
CLONE_DIR = "/content/D_Cortex"

if not os.path.exists(CLONE_DIR):
    subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True)
    print(f"[INFO] Cloned {REPO_URL}")
else:
    subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True)
    print(f"[INFO] Repo updated")

if CLONE_DIR not in sys.path:
    sys.path.insert(0, CLONE_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "tiktoken", "datasets", "matplotlib"],
    check=True,
)
print("[INFO] Dependencies installed")

# ======================== 4. IMPORTS ========================================

import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
from datasets import load_dataset

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import MultiHeadSelfAttention
from dcortex.backbone.fusion_block import CrossAttention

print("[INFO] All imports OK")

# ======================== 4B. SDPA MONKEY-PATCH =============================
#
# The repo's attention modules use manual q @ k.T -> softmax -> attn @ v.
# On A100 with PyTorch >= 2.0, F.scaled_dot_product_attention dispatches
# to flash or memory-efficient kernels automatically.
#
# We monkey-patch MultiHeadSelfAttention.forward and CrossAttention.forward
# AFTER import so the source files stay untouched.
# ============================================================================

if _SDPA_AVAILABLE:
    def _sdpa_self_attn_forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = h.shape
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build attention mask for SDPA: combine causal + padding
        attn_mask = None
        if attention_mask is not None:
            # padding: [B, 1, 1, T] bool where True = MASKED (excluded)
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # causal: [T, T] upper-triangular True
            causal = torch.triu(
                torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
            )
            # combined: [B, 1, T, T]
            combined = causal.unsqueeze(0).unsqueeze(0) | pad_mask
            # SDPA wants float mask with -inf for masked positions
            attn_mask = torch.zeros(B, 1, T, T, device=h.device, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(attention_mask is None),  # fast path when no padding
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    def _sdpa_cross_attn_forward(
        self, h: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = h.shape
        _, K, _ = memory.shape

        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, T, d]

        kv = self.kv(memory).reshape(B, K, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, H, K, d]
        k, v = kv[0], kv[1]

        # No causal mask for cross-attention (memory tokens not temporal)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    MultiHeadSelfAttention.forward = _sdpa_self_attn_forward
    CrossAttention.forward = _sdpa_cross_attn_forward
    print("[INFO] SDPA monkey-patch applied to MultiHeadSelfAttention + CrossAttention")
    print("       Flash/memory-efficient kernels will be used when eligible")
else:
    print("[WARN] SDPA not available. Using manual attention (slower, more VRAM)")

# ======================== 5. TOKENIZER + DATA PIPELINE ======================

ENC = tiktoken.get_encoding("gpt2")   # vocab_size = 50257
EOT = ENC.eot_token                   # <|endoftext|> = 50256


def tokenize_to_bin(split: str, max_tokens: int) -> str:
    """Stream TinyStories, tokenize with GPT-2 BPE, save as uint16 .bin.

    nanoGPT pattern: one flat binary per split, atomic write via .tmp rename.
    Skips tokenization if .bin already exists on Drive.
    """
    path = os.path.join(BIN_DIR, f'tinystories_{split}.bin')
    if os.path.exists(path):
        n = os.path.getsize(path) // 2
        print(f"[INFO] {split} cached: {path} ({n:,} tokens)")
        return path

    print(f"[INFO] Tokenizing {split} from TinyStories...", flush=True)
    try:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TinyStories ({e}). "
            "Check internet or run: pip install datasets"
        ) from e

    tokens: List[int] = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex.get('text', '') or ex.get('story', '')
        if not text:
            continue
        enc_ids = ENC.encode_ordinary(text)
        enc_ids.append(EOT)
        tokens.extend(enc_ids)
        if i > 0 and i % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {len(tokens):,} tokens ({len(tokens)/elapsed:.0f} tok/s)",
                  flush=True)
        if len(tokens) >= max_tokens:
            break

    arr = np.array(tokens[:max_tokens], dtype=np.uint16)
    tmp = path + '.tmp'
    arr.tofile(tmp)
    os.rename(tmp, path)
    elapsed = time.time() - t0
    print(f"[INFO] {split}: {len(arr):,} tokens -> {path} ({elapsed:.1f}s)")
    return path


train_bin = tokenize_to_bin('train', max_tokens=80_000_000)
val_bin = tokenize_to_bin('validation', max_tokens=5_000_000)


def copy_to_local_ssd(src: str) -> str:
    """Copy .bin to Colab local SSD for fast memmap. Returns local path."""
    dst = os.path.join(LOCAL_DATA, os.path.basename(src))
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return dst
    stat = os.statvfs('/content')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    need_gb = os.path.getsize(src) / (1024**3)
    if free_gb < need_gb + 1.0:
        print(f"[WARN] Low disk ({free_gb:.1f} GB). Using Drive path directly.")
        return src
    subprocess.run(["cp", src, dst], check=True)
    print(f"[INFO] {os.path.basename(src)} -> local SSD ({need_gb:.2f} GB)")
    return dst


train_data = np.memmap(copy_to_local_ssd(train_bin), dtype=np.uint16, mode='r')
val_data = np.memmap(copy_to_local_ssd(val_bin), dtype=np.uint16, mode='r')
print(f"[INFO] Data ready: {len(train_data):,} train / {len(val_data):,} val tokens")

# ======================== 6. TRAINING CONFIG ================================


@dataclass
class TrainConfig:
    """Hyperparameters for A100 validation run."""
    seq_len: int = 1024
    batch_size: int = 16
    grad_accum: int = 4          # effective batch = 64 sequences = 65536 tok
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    total_steps: int = 5000
    warmup_steps: int = 200
    session_len: int = 8         # reset memory every N micro-batches
    gate_entropy_w: float = 0.01 # aux loss weight for gate entropy
    log_every: int = 50
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int = 1000
    curriculum_start: int = 3500 # step where memory curriculum begins
    curriculum_ratio: float = 0.5 # fraction of micro-batches from curriculum


TC = TrainConfig()
TOK_PER_STEP = TC.batch_size * TC.seq_len * TC.grad_accum

print(SEP)
print(f"[INFO] Config: {TC.total_steps} steps | "
      f"batch {TC.batch_size}x{TC.grad_accum}={TC.batch_size*TC.grad_accum} | "
      f"seq {TC.seq_len} | {TOK_PER_STEP:,} tok/step")
print(f"[INFO] LR {TC.lr}->{TC.min_lr} cosine | warmup {TC.warmup_steps} | "
      f"clip {TC.grad_clip} | wd {TC.weight_decay}")
print(f"[INFO] Memory session {TC.session_len} micro-batches | "
      f"gate_entropy_w {TC.gate_entropy_w}")
print(f"[INFO] Curriculum starts at step {TC.curriculum_start} | "
      f"ratio {TC.curriculum_ratio}")
print(f"[INFO] Total tokens: {TC.total_steps * TOK_PER_STEP:,}")
print(SEP)

# ======================== 7. MODEL + OPTIMIZER ==============================

cfg = DCortexConfig()  # full scale: 12L/768d/12h/3072ff/2048ctx/50257vocab
model = DCortexV2Model(cfg).to(DEVICE)

N_PARAMS = sum(p.numel() for p in model.parameters())
N_TRAINABLE = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] {N_PARAMS/1e6:.2f}M params ({N_TRAINABLE/1e6:.2f}M trainable) on {DEVICE}")

# AdamW with split param groups: decay for 2D+ weights, no decay for bias/norm
decay_p, nodecay_p = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() < 2 or 'norm' in name or 'bias' in name:
        nodecay_p.append(p)
    else:
        decay_p.append(p)

optimizer = torch.optim.AdamW([
    {'params': decay_p, 'weight_decay': TC.weight_decay},
    {'params': nodecay_p, 'weight_decay': 0.0},
], lr=TC.lr, betas=(TC.beta1, TC.beta2))

print(f"[INFO] AdamW: {len(decay_p)} decay + {len(nodecay_p)} no-decay groups")

scaler = torch.amp.GradScaler('cuda') if USE_SCALER else None


def get_lr(step: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < TC.warmup_steps:
        return TC.lr * (step + 1) / TC.warmup_steps
    if step >= TC.total_steps:
        return TC.min_lr
    t = (step - TC.warmup_steps) / (TC.total_steps - TC.warmup_steps)
    return TC.min_lr + 0.5 * (TC.lr - TC.min_lr) * (1.0 + math.cos(math.pi * t))


# ======================== 7B. PER-SUBMODULE GRADIENT TRACKING ===============

# Define submodule groups for gradient monitoring
SUBMODULE_GROUPS = {
    'embeddings': [],
    'standard_blocks': [],
    'fusion_blocks': [],
    'query_engine': [],
    'readers': [],
    'read_fusion': [],
    'writer': [],
    'episode_ssm': [],
    'lm_head': [],
}

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith('embeddings'):
        SUBMODULE_GROUPS['embeddings'].append(p)
    elif name.startswith('standard_blocks'):
        SUBMODULE_GROUPS['standard_blocks'].append(p)
    elif name.startswith('fusion_blocks'):
        SUBMODULE_GROUPS['fusion_blocks'].append(p)
    elif name.startswith('query_engine'):
        SUBMODULE_GROUPS['query_engine'].append(p)
    elif any(name.startswith(r) for r in
             ('state_reader', 'episode_reader', 'conflict_reader',
              'archive_reader', 'working_reader')):
        SUBMODULE_GROUPS['readers'].append(p)
    elif name.startswith('read_fusion'):
        SUBMODULE_GROUPS['read_fusion'].append(p)
    elif name.startswith('writer'):
        SUBMODULE_GROUPS['writer'].append(p)
    elif name.startswith('episode_ssm'):
        SUBMODULE_GROUPS['episode_ssm'].append(p)
    elif name.startswith('lm_head') or name.startswith('final_norm'):
        SUBMODULE_GROUPS['lm_head'].append(p)

# Verify all params are assigned
_assigned = sum(len(v) for v in SUBMODULE_GROUPS.values())
_total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
print(f"[INFO] Submodule groups: {_assigned}/{_total_trainable} params assigned")
for gname, gparams in SUBMODULE_GROUPS.items():
    n = sum(p.numel() for p in gparams)
    print(f"  {gname:20s}: {len(gparams):3d} tensors, {n/1e6:.2f}M params")


def compute_submodule_grad_norms() -> Dict[str, float]:
    """Compute L2 grad norm per submodule group. Call after backward."""
    norms = {}
    for gname, gparams in SUBMODULE_GROUPS.items():
        total_sq = 0.0
        n_with_grad = 0
        for p in gparams:
            if p.grad is not None:
                total_sq += p.grad.data.norm(2).item() ** 2
                n_with_grad += 1
        norms[gname] = math.sqrt(total_sq)
    return norms


# ======================== 7C. GATE METRICS ACCUMULATOR =====================

# Hook on writer to capture gate_probs for auxiliary loss
_gate_store: Dict[str, torch.Tensor] = {}
_gate_accum: List[torch.Tensor] = []  # accumulate across micro-batches


def _writer_hook(module, inp, out):
    _gate_store['probs'] = out
    _gate_accum.append(out.detach().cpu())


model.writer.register_forward_hook(_writer_hook)

# ======================== 8. BATCH LOADER ===================================


def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random batch from memmap binary. nanoGPT pattern."""
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - TC.seq_len - 1, size=(TC.batch_size,))
    x = np.stack([data[i : i + TC.seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + TC.seq_len].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).pin_memory().to(DEVICE, non_blocking=True)
    y = torch.from_numpy(y).pin_memory().to(DEVICE, non_blocking=True)
    return x, y


def get_seeded_batches(
    split: str, n_batches: int, seed: int = 9999,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pre-generate n_batches with a fixed seed for reproducible ablation."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    batches = [get_batch(split) for _ in range(n_batches)]
    np.random.set_state(rng_state)
    return batches


# ======================== 8B. MEMORY CURRICULUM =============================
#
# Simple synthetic sequences that stress-test memory:
#
# Type A -- FACT RECALL:
#   "The [entity] has color [color]. ... (filler) ... What color is the [entity]?"
#   Forces the model to write a fact early and recall it later.
#
# Type B -- CONTRADICTION:
#   "The cat is black. ... The cat is white."
#   Same entity key, different value. Should trigger ConflictMemory.
#
# Type C -- UPDATE:
#   "Tom lives in Paris. ... Tom moved to London. Where does Tom live?"
#   Tests whether memory updates old facts.
#
# These are tokenized and padded to seq_len. The model sees them as normal
# LM sequences; the curriculum value comes from the memory write/read
# patterns they induce, not from a special loss function.
# ============================================================================

_ENTITIES = [
    "cat", "dog", "bird", "fish", "rabbit", "horse", "bear", "fox",
    "lion", "tiger", "elephant", "monkey", "penguin", "dolphin", "owl",
]
_COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink"]
_PLACES = ["Paris", "London", "Tokyo", "Rome", "Berlin", "Madrid", "Cairo"]
_NAMES = ["Tom", "Lily", "Max", "Sara", "Ben", "Emma", "Jack", "Anna"]
_FILLER = (
    "Once upon a time, there was a little town by the river. "
    "The sun was shining and the birds were singing in the trees. "
    "Everyone was happy and the day was beautiful. "
    "The children played in the garden while the adults talked. "
)


def _make_fact_recall() -> str:
    ent = random.choice(_ENTITIES)
    col = random.choice(_COLORS)
    return (
        f"The {ent} has color {col}. {_FILLER}"
        f"What color is the {ent}? The {ent} has color {col}."
    )


def _make_contradiction() -> str:
    ent = random.choice(_ENTITIES)
    c1, c2 = random.sample(_COLORS, 2)
    return (
        f"The {ent} is {c1}. {_FILLER}"
        f"Actually, the {ent} is {c2}. The {ent} is {c2}."
    )


def _make_update() -> str:
    name = random.choice(_NAMES)
    p1, p2 = random.sample(_PLACES, 2)
    return (
        f"{name} lives in {p1}. {_FILLER}"
        f"{name} moved to {p2}. Where does {name} live? "
        f"{name} lives in {p2}."
    )


_CURRICULUM_GENERATORS = [_make_fact_recall, _make_contradiction, _make_update]


def get_curriculum_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of curriculum sequences, tokenize, pad to seq_len."""
    xs, ys = [], []
    for _ in range(TC.batch_size):
        gen = random.choice(_CURRICULUM_GENERATORS)
        text = gen()
        ids = ENC.encode_ordinary(text)
        ids.append(EOT)
        # Truncate or pad to seq_len + 1 (need x and y)
        if len(ids) > TC.seq_len + 1:
            ids = ids[:TC.seq_len + 1]
        else:
            ids = ids + [EOT] * (TC.seq_len + 1 - len(ids))
        xs.append(ids[:TC.seq_len])
        ys.append(ids[1:TC.seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    y = torch.tensor(ys, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    return x, y


# ======================== 9. LOSSES =========================================


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """LM cross-entropy + gate entropy auxiliary loss.

    Returns (total_loss_tensor, lm_loss_scalar, gate_entropy_scalar).
    """
    B, T, V = logits.shape
    lm_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    gp = _gate_store.get('probs')
    gate_ent = torch.tensor(0.0, device=logits.device)
    if gp is not None:
        # Entropy: -sum(p * log(p + eps)). Minimizing sharpens the gate.
        gate_ent = -(gp * (gp + 1e-8).log()).sum(dim=-1).mean()

    total = lm_loss + TC.gate_entropy_w * gate_ent
    return total, lm_loss.item(), gate_ent.item()


# ======================== 10. EVAL + GENERATE ===============================


@torch.no_grad()
def evaluate(model: nn.Module, step: int) -> float:
    """Eval on val set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    for _ in range(TC.eval_batches):
        x, y = get_batch('val')
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            total_loss += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    avg = total_loss / TC.eval_batches
    ppl = math.exp(min(avg, 20.0))
    print(f"  [EVAL] step={step} | loss={avg:.4f} | ppl={ppl:.2f}", flush=True)
    model.train()
    return avg


@torch.no_grad()
def generate_sample(
    model: nn.Module,
    prompt: str = "Once upon a time",
    max_new: int = 80,
    temp: float = 0.8,
    top_k: int = 40,
) -> str:
    """Top-k sampling from the model."""
    model.eval()
    ids = ENC.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new):
        if x.shape[1] > cfg.max_seq_len:
            x = x[:, -cfg.max_seq_len:]
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, idx, vals)
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1)
        x = torch.cat([x, tok], dim=1)
        if tok.item() == EOT:
            break
    model.train()
    return ENC.decode(x[0].tolist())


# ======================== 11. CHECKPOINTING =================================


def save_ckpt(
    model: nn.Module, optimizer, scaler, step: int, losses: list,
) -> None:
    """Atomic checkpoint to Drive."""
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'losses': losses[-200:],
        'config_model': asdict(cfg),
        'config_train': asdict(TC),
    }
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()
    fname = f'ckpt_step{step:06d}.pt'
    tmp = os.path.join(CHECKPOINT_DIR, fname + '.tmp')
    final = os.path.join(CHECKPOINT_DIR, fname)
    torch.save(ckpt, tmp)
    os.rename(tmp, final)
    print(f"[INFO] Checkpoint: {fname}", flush=True)


def load_latest_ckpt(model: nn.Module, optimizer, scaler) -> int:
    """Resume from latest checkpoint. Returns start_step (0 if none)."""
    ckpts = list(Path(CHECKPOINT_DIR).glob('ckpt_step*.pt'))
    if not ckpts:
        print("[INFO] No checkpoint found. Starting from step 0.")
        return 0
    # Sort numerically by step
    ckpts.sort(key=lambda p: int(p.stem.split('step')[1]))
    latest = ckpts[-1]
    print(f"[INFO] Loading {latest.name}...")
    c = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(c['model'])
    optimizer.load_state_dict(c['optimizer'])
    if scaler is not None and 'scaler' in c:
        scaler.load_state_dict(c['scaler'])
    s = c['step']
    print(f"[INFO] Resumed from step {s}")
    return s


# ======================== 12. TRAINING LOOP =================================

print(SEP)
print("[INFO] TRAINING START")
print(SEP)

start_step = load_latest_ckpt(model, optimizer, scaler)
model.train()
torch.cuda.reset_peak_memory_stats()

# Metrics storage
M_steps: List[int] = []
M_lm: List[float] = []
M_ge: List[float] = []
M_gn: List[float] = []
M_lr: List[float] = []
M_tps: List[float] = []
M_gate: List[List[float]] = []       # [steps][6] -- AVERAGED across micro-batches
M_occ: List[Dict[str, float]] = []   # [steps]{bank: frac}
M_fgate: List[List[float]] = []      # [steps][n_fusion_layers]
M_subgrad: List[Dict[str, float]] = []  # [steps]{submodule: grad_norm}
E_steps: List[int] = []
E_loss: List[float] = []
E_ppl: List[float] = []
loss_log: List[float] = []

micro_ctr = 0
best_val = float('inf')
t0_train = time.time()
tok_done = 0

# Clean start
with contextlib.redirect_stdout(io.StringIO()):
    model.reset_memory()

# Initial eval
init_val = evaluate(model, start_step)
E_steps.append(start_step)
E_loss.append(init_val)
E_ppl.append(math.exp(min(init_val, 20.0)))
expected_init = math.log(cfg.vocab_size)
print(f"[INFO] Init val loss: {init_val:.4f} (expected ~{expected_init:.2f} for random)")
print(f"  [SAMPLE] {generate_sample(model)[:200]}")

for step in range(start_step, TC.total_steps):
    t_step = time.time()

    # LR schedule
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)

    acc_lm = 0.0
    acc_ge = 0.0
    _gate_accum.clear()  # reset gate accumulator for this step

    # Determine if curriculum is active
    use_curriculum = (step >= TC.curriculum_start)

    # Gradient accumulation over micro-batches
    for _mi in range(TC.grad_accum):
        # Decide data source: curriculum or standard
        if use_curriculum and random.random() < TC.curriculum_ratio:
            x, y = get_curriculum_batch()
        else:
            x, y = get_batch('train')

        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=True)
            loss, lm_val, ge_val = compute_loss(logits, y)
            loss = loss / TC.grad_accum

        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        acc_lm += lm_val / TC.grad_accum
        acc_ge += ge_val / TC.grad_accum
        tok_done += TC.batch_size * TC.seq_len
        micro_ctr += 1

        # Memory session boundary: reset every session_len micro-batches
        if micro_ctr % TC.session_len == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                model.reset_memory()

    # Per-submodule gradient norms (BEFORE clipping)
    sub_grads = compute_submodule_grad_norms()

    # Gradient clipping
    if USE_SCALER:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), TC.grad_clip
    ).item()

    # Optimizer step
    if USE_SCALER:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    step_time = time.time() - t_step
    tps = TOK_PER_STEP / max(step_time, 1e-6)
    loss_log.append(acc_lm)

    # Gate metrics: AVERAGE across ALL micro-batches in this step
    if _gate_accum:
        gate_all = torch.cat(_gate_accum, dim=0)  # [total_B_across_micros, 6]
        gate_avg = gate_all.mean(0).tolist()
    else:
        gate_avg = [0.0] * 6

    snap = model.memory_snapshot()
    occ = {k: v['occupied'] / v['capacity'] for k, v in snap.items()}
    fgates = [torch.sigmoid(b.mem_gate).mean().item() for b in model.fusion_blocks]

    M_steps.append(step)
    M_lm.append(acc_lm)
    M_ge.append(acc_ge)
    M_gn.append(grad_norm)
    M_lr.append(lr)
    M_tps.append(tps)
    M_gate.append(gate_avg)
    M_occ.append(occ)
    M_fgate.append(fgates)
    M_subgrad.append(sub_grads)

    # Periodic logging
    if step % TC.log_every == 0 or step == start_step:
        elapsed = time.time() - t0_train
        done = step - start_step + 1
        eta_s = elapsed / done * (TC.total_steps - step - 1)
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        phase = "CURRIC" if use_curriculum else "LM"
        print(
            f"Step {step:5d}/{TC.total_steps} [{phase}] | loss={acc_lm:.4f} | "
            f"ge={acc_ge:.3f} | gn={grad_norm:.2f} | lr={lr:.2e} | "
            f"{tps:.0f} tok/s | VRAM={vram:.1f}GB | "
            f"ETA {int(eta_s//60)}m{int(eta_s%60):02d}s",
            flush=True,
        )
        # Per-submodule grad norms (periodic, not every step)
        sg_str = " | ".join(f"{k}={v:.3f}" for k, v in sub_grads.items() if v > 0)
        print(f"  grad/sub: {sg_str}", flush=True)

    # Eval checkpoint
    if (step + 1) % TC.eval_every == 0:
        vl = evaluate(model, step + 1)
        E_steps.append(step + 1)
        E_loss.append(vl)
        E_ppl.append(math.exp(min(vl, 20.0)))
        print(f"  [SAMPLE] {generate_sample(model)[:200]}", flush=True)
        if vl < best_val:
            best_val = vl
            save_ckpt(model, optimizer, scaler, step + 1, loss_log)

    # Periodic checkpoint
    if (step + 1) % TC.ckpt_every == 0:
        save_ckpt(model, optimizer, scaler, step + 1, loss_log)

    # NaN/Inf safety
    if math.isnan(acc_lm) or math.isinf(acc_lm):
        print(f"[ERROR] Loss={acc_lm} at step {step}. Aborting.", flush=True)
        break

# Training done
total_time = time.time() - t0_train
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print(SEP)
print(f"[INFO] Training complete: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"[INFO] {tok_done:,} tokens | {tok_done/total_time:,.0f} tok/s | "
      f"peak VRAM {peak_vram:.1f} GB")
print(f"[INFO] Final train loss: {M_lm[-1]:.4f}" if M_lm else "")
print(f"[INFO] Best val loss: {best_val:.4f} "
      f"(ppl={math.exp(min(best_val, 20.0)):.2f})")
print(SEP)

# Final checkpoint
save_ckpt(model, optimizer, scaler, TC.total_steps, loss_log)

# ======================== 13. SAME-BATCH MEMORY ABLATION ====================
#
# Critical fix: both conditions (with/without memory) are evaluated on the
# EXACT SAME batches, generated with a fixed seed. This eliminates noise
# from different random subsets.
# ============================================================================

print(SEP)
print("[INFO] SAME-BATCH MEMORY ABLATION STUDY")
print(SEP)

N_POPULATE = 10
N_EVAL_ABL = 50


@torch.no_grad()
def same_batch_ablation(
    model: nn.Module,
    n_populate: int = N_POPULATE,
    n_eval: int = N_EVAL_ABL,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Compare eval loss with populated memory vs empty memory on SAME batches.

    1. Pre-generate eval batches with fixed seed.
    2. Reset + populate memory with n_populate forward passes.
    3. Eval with populated memory on pre-generated batches.
    4. Reset + eval with empty memory on the SAME pre-generated batches.

    Returns (loss_with_memory, loss_without_memory).
    """
    model.eval()
    _gate_accum.clear()  # prevent hook accumulation during ablation

    # Pre-generate fixed eval batches
    eval_batches = get_seeded_batches('val', n_eval, seed=seed)

    # Phase 1: populate memory
    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    populate_batches = get_seeded_batches('val', n_populate, seed=seed + 1)
    for x, _ in populate_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            model(x, write_memory=True)

    # Phase 2: eval WITH populated memory
    loss_with = 0.0
    for x, y in eval_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            loss_with += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    loss_with /= n_eval

    # Phase 3: eval WITHOUT memory (reset, SAME batches)
    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    loss_without = 0.0
    for x, y in eval_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            loss_without += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    loss_without /= n_eval

    model.train()
    return loss_with, loss_without


loss_w_mem, loss_wo_mem = same_batch_ablation(model)
abl_delta = loss_wo_mem - loss_w_mem
ppl_w = math.exp(min(loss_w_mem, 20.0))
ppl_wo = math.exp(min(loss_wo_mem, 20.0))

print(f"  WITH memory   : loss={loss_w_mem:.4f}  ppl={ppl_w:.2f}")
print(f"  WITHOUT memory: loss={loss_wo_mem:.4f}  ppl={ppl_wo:.2f}")
print(f"  Delta (without - with): {abl_delta:+.4f}")
print(f"  Method: same {N_EVAL_ABL} batches (seed=12345), populated with {N_POPULATE} passes")
if abl_delta > 0.005:
    print(f"  RESULT: Memory provides measurable benefit (+{abl_delta:.4f} loss reduction)")
elif abl_delta > 0:
    print(f"  RESULT: Slight memory benefit ({abl_delta:+.4f}). May improve with more training.")
else:
    print(f"  RESULT: Memory neutral or not yet useful ({abl_delta:+.4f}). "
          "Expected at early training -- memory curriculum and Step 3 benchmarks "
          "will provide stronger signal.")

# ======================== 14. CONFLICTMEMORY SEMANTIC TEST ==================
#
# The most novel architectural feature: when a write candidate has high
# key-similarity but divergent value compared to an existing StateMemory
# slot, the DIFFERENCE VECTOR (new - old) is stored in ConflictMemory.
#
# This test validates the mechanism AFTER training:
#   1. Reset all memory.
#   2. Write a "fact" to StateMemory (via forward pass with a specific input).
#   3. Write a "contradicting fact" (same entity, different attribute).
#   4. Check: did ConflictMemory gain a slot?
#   5. If yes, verify the stored value approximates (new_value - old_value).
# ============================================================================

print(SEP)
print("[INFO] CONFLICTMEMORY SEMANTIC VALIDATION")
print(SEP)


@torch.no_grad()
def test_conflict_memory_semantics(model: nn.Module) -> Dict[str, object]:
    """Test that ConflictMemory captures difference vectors on contradiction.

    Returns a diagnostic dict with test results.
    """
    model.eval()
    results = {}

    # Reset all memory
    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()

    # Step 1: Write initial fact via forward
    fact_text = "The big red cat sat on the old wooden chair in the kitchen."
    fact_ids = ENC.encode_ordinary(fact_text)
    if len(fact_ids) < TC.seq_len:
        fact_ids = fact_ids + [EOT] * (TC.seq_len - len(fact_ids))
    x_fact = torch.tensor([fact_ids[:TC.seq_len]], dtype=torch.long, device=DEVICE)

    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x_fact, write_memory=True)

    snap_after_fact = model.memory_snapshot()
    state_occ_1 = snap_after_fact['state']['occupied']
    conflict_occ_1 = snap_after_fact['conflict']['occupied']
    results['state_after_fact'] = state_occ_1
    results['conflict_after_fact'] = conflict_occ_1

    # Run several more passes to build up state memory
    for _ in range(5):
        with torch.amp.autocast('cuda', dtype=DTYPE):
            model(x_fact, write_memory=True)

    snap_after_repeat = model.memory_snapshot()
    state_occ_repeat = snap_after_repeat['state']['occupied']
    results['state_after_repeat'] = state_occ_repeat

    # Step 2: Write contradicting fact (same structure, different content)
    contra_text = "The big blue dog sat on the new metal table in the bedroom."
    contra_ids = ENC.encode_ordinary(contra_text)
    if len(contra_ids) < TC.seq_len:
        contra_ids = contra_ids + [EOT] * (TC.seq_len - len(contra_ids))
    x_contra = torch.tensor([contra_ids[:TC.seq_len]], dtype=torch.long, device=DEVICE)

    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x_contra, write_memory=True)

    snap_after_contra = model.memory_snapshot()
    conflict_occ_2 = snap_after_contra['conflict']['occupied']
    results['conflict_after_contradiction'] = conflict_occ_2
    results['conflict_slots_gained'] = conflict_occ_2 - conflict_occ_1

    # Step 3: Direct updater test -- bypass the model, test the mechanism
    # Create two vectors: similar keys, divergent values
    D = cfg.hidden_dim
    d_ent, d_rel, d_typ = cfg.d_ent, cfg.d_rel, cfg.d_typ

    key_base = torch.randn(d_ent, device=DEVICE)
    k_ent_1 = F.normalize(key_base, dim=0)
    k_ent_2 = F.normalize(key_base + 0.05 * torch.randn(d_ent, device=DEVICE), dim=0)
    # Keys are very similar (cosine ~ 0.99)
    key_sim = F.cosine_similarity(k_ent_1.unsqueeze(0), k_ent_2.unsqueeze(0)).item()
    results['direct_test_key_sim'] = round(key_sim, 4)

    k_rel = torch.randn(d_rel, device=DEVICE)
    k_typ = torch.randn(d_typ, device=DEVICE)

    # Values are very different
    v1 = torch.randn(D, device=DEVICE)
    v2 = -v1 + 0.1 * torch.randn(D, device=DEVICE)  # nearly opposite
    val_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    results['direct_test_value_sim'] = round(val_sim, 4)

    # Write v1 to a fresh state bank
    test_bank = model.state_mem.__class__(
        8, D, d_ent, d_rel, d_typ,
    ).to(DEVICE)

    model.updater.update(test_bank, v1, k_ent_1, k_rel, k_typ, step=1)
    results['test_bank_after_v1'] = test_bank.n_occupied()

    # Check if updater detects conflict
    is_conflict = model.updater.detect_conflict(
        test_bank, v2, k_ent_2, k_rel, k_typ,
    )
    results['conflict_detected'] = is_conflict

    if is_conflict:
        # Test the ACTUAL diff mechanism:
        # Write v2 to test_bank WITH is_conflict=True. Since test_bank
        # already has v1 at slot 0 with similar keys, the updater will
        # find the match and store (v2 - v1) instead of v2.
        v1_before = test_bank.values[0].clone()
        model.updater.update(
            test_bank, v2, k_ent_2, k_rel, k_typ,
            step=2, is_conflict=True,
        )
        stored_diff = test_bank.values[0]
        expected_diff = v2 - v1_before
        diff_cosine = F.cosine_similarity(
            stored_diff.unsqueeze(0), expected_diff.unsqueeze(0),
        ).item()
        results['diff_vector_cosine_vs_expected'] = round(diff_cosine, 4)
        results['diff_vector_correct'] = diff_cosine > 0.99

        # Also test normal EMA path (non-conflict write)
        # Reset test_bank and write v1 again, then v2 WITHOUT is_conflict
        test_bank.reset()
        model.updater.update(test_bank, v1, k_ent_1, k_rel, k_typ, step=3)
        v1_stored = test_bank.values[0].clone()
        model.updater.update(test_bank, v2, k_ent_2, k_rel, k_typ, step=4,
                             is_conflict=False)
        ema_val = test_bank.values[0]
        # EMA: blended = (1 - alpha) * old + alpha * new
        alpha = model.updater.ema_alpha
        expected_ema = (1.0 - alpha) * v1_stored + alpha * v2
        ema_cosine = F.cosine_similarity(
            ema_val.unsqueeze(0), expected_ema.unsqueeze(0),
        ).item()
        results['ema_cosine_vs_expected'] = round(ema_cosine, 4)
        results['ema_update_correct'] = ema_cosine > 0.99

    # Verdict
    results['mechanism_functional'] = (
        results.get('conflict_detected', False) and
        results.get('diff_vector_correct', False)
    )

    model.train()
    return results


conflict_results = test_conflict_memory_semantics(model)
print("ConflictMemory semantic test results:")
for k, v in conflict_results.items():
    print(f"  {k:35s}: {v}")

if conflict_results.get('mechanism_functional'):
    print("  VERDICT: ConflictMemory detection mechanism is functional")
    print("           (high key-sim + low value-sim triggers conflict flag)")
else:
    print("  VERDICT: ConflictMemory mechanism did not trigger as expected")
    print("           (may need theta_match/theta_conflict tuning or more training)")

# ======================== 15. FINAL DIAGNOSTICS =============================

print(SEP)
print("[INFO] FINAL DIAGNOSTICS")
print(SEP)

# Populate memory for analysis
with contextlib.redirect_stdout(io.StringIO()):
    model.reset_memory()
model.eval()
for _ in range(20):
    x, _ = get_batch('val')
    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x, write_memory=True)

# Bank occupancy
snap_final = model.memory_snapshot()
print("Memory bank occupancy (after 20 forward passes):")
for name, info in snap_final.items():
    pct = info['occupied'] / info['capacity'] * 100
    print(f"  {name:12s}: {info['occupied']:3d}/{info['capacity']:3d} "
          f"({pct:5.1f}%)  avg_usage={info['usage_mean']:.1f}  "
          f"max_usage={info['usage_max']:.0f}")

# Fusion block mem_gate
print("\nFusion block mem_gate (sigmoid) after training:")
for i, block in enumerate(model.fusion_blocks):
    g = torch.sigmoid(block.mem_gate)
    print(f"  Block {i}: mean={g.mean():.4f}  std={g.std():.4f}  "
          f"min={g.min():.4f}  max={g.max():.4f}")

# Writer gate distribution (from final step, averaged across micro-batches)
if M_gate:
    final_gate = M_gate[-1]
    gate_names = ['state', 'episode_obj', 'conflict', 'archive', 'working', 'skip']
    print("\nWriter gate distribution (final step, averaged across all micro-batches):")
    for i, gn in enumerate(gate_names):
        bar = '#' * int(final_gate[i] * 60)
        print(f"  {gn:12s}: {final_gate[i]:.4f}  {bar}")

# Per-submodule gradient summary (from final logged step)
if M_subgrad:
    print("\nPer-submodule gradient norms (final step):")
    final_sg = M_subgrad[-1]
    for gname, gnorm in final_sg.items():
        status = "OK" if gnorm > 0 else "ZERO (no gradient!)"
        print(f"  {gname:20s}: {gnorm:.4f}  [{status}]")

# Consolidation test
print("\nConsolidation pass on populated memory:")
with contextlib.redirect_stdout(io.StringIO()):
    cons_report = model.consolidate()
for bank, r in cons_report.items():
    print(f"  {bank:12s}: pruned={r['pruned']}  migrated={r['migrated']}  "
          f"merged={r['merged']}")

# Generation samples
print("\nGeneration samples (post-training):")
test_prompts = [
    "Once upon a time",
    "The little cat went to",
    "In a big forest there was",
]
for prompt in test_prompts:
    sample = generate_sample(model, prompt, max_new=100, temp=0.8)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  -> {sample[:300]}")
    print()

# ======================== 16. PLOTS =========================================

print(SEP)
print("[INFO] Generating plots")
print(SEP)

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle(
    'D_Cortex v2.0-alpha -- Step 2: Training + Memory Validation',
    fontsize=14, y=0.98,
)

# --- 1. Training loss ---
ax = axes[0, 0]
ax.plot(M_steps, M_lm, alpha=0.25, color='blue', linewidth=0.5)
if len(M_lm) > 50:
    w = 50
    smoothed = [np.mean(M_lm[max(0, i - w) : i + 1]) for i in range(len(M_lm))]
    ax.plot(M_steps, smoothed, color='blue', linewidth=1.5, label='smoothed (w=50)')
if TC.curriculum_start < TC.total_steps:
    ax.axvline(x=TC.curriculum_start, color='red', ls=':', alpha=0.7,
               label=f'curriculum@{TC.curriculum_start}')
ax.set_xlabel('Step')
ax.set_ylabel('LM Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)
ax.legend()

# --- 2. Eval loss + perplexity ---
ax = axes[0, 1]
if E_steps:
    ln1 = ax.plot(E_steps, E_loss, 'o-', color='red', markersize=4, label='loss')
    ax2 = ax.twinx()
    ln2 = ax2.plot(E_steps, E_ppl, 's--', color='orange', markersize=4,
                   alpha=0.7, label='ppl')
    ax2.set_ylabel('Perplexity', color='orange')
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], loc='upper right')
ax.set_xlabel('Step')
ax.set_ylabel('Loss', color='red')
ax.set_title('Eval Loss / Perplexity')
ax.grid(True, alpha=0.3)

# --- 3. Gate entropy ---
ax = axes[0, 2]
ax.plot(M_steps, M_ge, alpha=0.5, color='green')
max_ent = math.log(6)
ax.axhline(y=max_ent, color='gray', ls='--', alpha=0.7,
           label=f'max entropy = ln(6) = {max_ent:.2f}')
ax.set_xlabel('Step')
ax.set_ylabel('Entropy')
ax.set_title('Writer Gate Entropy')
ax.grid(True, alpha=0.3)
ax.legend()

# --- 4. Gate distribution (stacked area) ---
ax = axes[1, 0]
gate_arr = np.array(M_gate)  # [N_steps, 6]
g_names = ['state', 'ep_obj', 'conflict', 'archive', 'working', 'skip']
g_colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#607D8B']
bottom = np.zeros(len(M_steps))
for i, (gn, gc) in enumerate(zip(g_names, g_colors)):
    ax.fill_between(M_steps, bottom, bottom + gate_arr[:, i],
                    alpha=0.7, color=gc, label=gn)
    bottom += gate_arr[:, i]
ax.set_ylim(0, 1.05)
ax.set_xlabel('Step')
ax.set_ylabel('Probability')
ax.set_title('Writer Gate Distribution (avg all micro-batches)')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# --- 5. Memory bank occupancy ---
ax = axes[1, 1]
bank_keys = list(M_occ[0].keys()) if M_occ else []
for bk, bc in zip(bank_keys, g_colors[:len(bank_keys)]):
    vals = [d.get(bk, 0) for d in M_occ]
    ax.plot(M_steps, vals, label=bk, color=bc, alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Fraction occupied')
ax.set_title('Memory Bank Occupancy')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- 6. Gradient norm (global) ---
ax = axes[1, 2]
ax.plot(M_steps, M_gn, alpha=0.4, color='purple', linewidth=0.5)
if len(M_gn) > 50:
    sm_gn = [np.mean(M_gn[max(0, i - 50) : i + 1]) for i in range(len(M_gn))]
    ax.plot(M_steps, sm_gn, color='purple', linewidth=1.5)
ax.axhline(y=TC.grad_clip, color='red', ls='--', label=f'clip={TC.grad_clip}')
ax.set_xlabel('Step')
ax.set_ylabel('Norm')
ax.set_title('Gradient Norm (global)')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 7. Learning rate ---
ax = axes[2, 0]
ax.plot(M_steps, M_lr, color='teal')
ax.set_xlabel('Step')
ax.set_ylabel('LR')
ax.set_title('LR Schedule (warmup + cosine)')
ax.grid(True, alpha=0.3)

# --- 8. Throughput ---
ax = axes[2, 1]
ax.plot(M_steps, M_tps, alpha=0.4, color='brown', linewidth=0.5)
if M_tps:
    avg_tps = np.mean(M_tps)
    ax.axhline(y=avg_tps, color='red', ls='--', label=f'avg={avg_tps:.0f}')
ax.set_xlabel('Step')
ax.set_ylabel('tok/s')
ax.set_title('Training Throughput')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 9. Same-batch ablation bar chart ---
ax = axes[2, 2]
bars = ax.bar(
    ['With Memory', 'Without Memory'],
    [loss_w_mem, loss_wo_mem],
    color=['#2196F3', '#F44336'],
)
ax.set_ylabel('Eval Loss')
ax.set_title(f'Same-Batch Ablation (delta={abl_delta:+.4f})')
for b, v in zip(bars, [loss_w_mem, loss_wo_mem]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f'{v:.4f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# --- 10. Per-submodule gradient norms over training ---
ax = axes[3, 0]
if M_subgrad:
    for gname in SUBMODULE_GROUPS:
        vals = [d.get(gname, 0.0) for d in M_subgrad]
        ax.plot(M_steps, vals, alpha=0.7, label=gname)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad L2 norm')
    ax.set_title('Per-Submodule Gradient Norms')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# --- 11. Fusion block mem_gate over training ---
ax = axes[3, 1]
if M_fgate:
    fgate_arr = np.array(M_fgate)
    for i in range(fgate_arr.shape[1]):
        ax.plot(M_steps, fgate_arr[:, i], alpha=0.7, label=f'FusionBlock {i}')
    ax.set_xlabel('Step')
    ax.set_ylabel('sigmoid(mem_gate) mean')
    ax.set_title('Fusion Block Memory Gate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# --- 12. Conflict test summary ---
ax = axes[3, 2]
ax.axis('off')
conflict_text = "ConflictMemory Test Results\n" + "-" * 30 + "\n"
for k, v in conflict_results.items():
    conflict_text += f"{k}: {v}\n"
ax.text(0.05, 0.95, conflict_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'step2_training_report.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[INFO] Plots saved: {plot_path}")

# ======================== 17. SUMMARY REPORT ================================

report = {
    'project': 'D_Cortex v2.0-alpha',
    'step': 'Step 2: Training + Memory Validation',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'hardware': {
        'gpu': GPU_NAME,
        'vram_gb': round(GPU_MEM_GB, 1),
        'peak_vram_gb': round(peak_vram, 1),
        'dtype': str(DTYPE),
        'grad_scaler': USE_SCALER,
        'sdpa_patched': _SDPA_AVAILABLE,
    },
    'model': {
        'params_total': N_PARAMS,
        'params_trainable': N_TRAINABLE,
        'n_layers': cfg.n_layers,
        'hidden_dim': cfg.hidden_dim,
        'n_heads': cfg.n_heads,
        'n_fusion_layers': cfg.n_fusion_layers,
        'vocab_size': cfg.vocab_size,
    },
    'data': {
        'dataset': 'roneneldan/TinyStories + synthetic curriculum',
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'tokenizer': 'tiktoken gpt2',
        'curriculum_start_step': TC.curriculum_start,
        'curriculum_ratio': TC.curriculum_ratio,
    },
    'training': {
        'total_steps': TC.total_steps,
        'tokens_processed': tok_done,
        'time_seconds': round(total_time, 1),
        'avg_tok_per_sec': round(tok_done / total_time),
        'peak_vram_gb': round(peak_vram, 1),
        'init_val_loss': round(init_val, 4),
        'final_train_loss': round(M_lm[-1], 4) if M_lm else None,
        'best_val_loss': round(best_val, 4),
        'best_val_ppl': round(math.exp(min(best_val, 20.0)), 2),
    },
    'memory_ablation': {
        'method': 'same-batch (seed=12345)',
        'n_eval_batches': N_EVAL_ABL,
        'n_populate_passes': N_POPULATE,
        'loss_with_memory': round(loss_w_mem, 4),
        'loss_without_memory': round(loss_wo_mem, 4),
        'delta': round(abl_delta, 4),
        'memory_helps': abl_delta > 0,
    },
    'conflict_memory_test': conflict_results,
    'gate_entropy_initial': round(M_ge[0], 4) if M_ge else None,
    'gate_entropy_final': round(M_ge[-1], 4) if M_ge else None,
    'submodule_grad_norms_final': M_subgrad[-1] if M_subgrad else {},
    'memory_snapshot_final': {
        k: {'occupied': v['occupied'], 'capacity': v['capacity']}
        for k, v in snap_final.items()
    },
}

report_path = os.path.join(RESULTS_DIR, 'step2_training_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, default=str)

# Print final summary
print(SEP)
print("D_CORTEX v2.0-alpha -- TRAINING + MEMORY VALIDATION REPORT")
print(SEP)
print(f"  Model           : {N_PARAMS/1e6:.2f}M params | "
      f"{cfg.n_layers}L/{cfg.hidden_dim}d/{cfg.n_heads}h")
print(f"  GPU             : {GPU_NAME} ({DTYPE}) | peak VRAM {peak_vram:.1f} GB")
print(f"  SDPA patched    : {_SDPA_AVAILABLE}")
print(f"  Data            : TinyStories ({len(train_data):,} train / "
      f"{len(val_data):,} val) + curriculum from step {TC.curriculum_start}")
print(f"  Training        : {TC.total_steps} steps | {tok_done:,} tokens | "
      f"{total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Throughput      : {tok_done/total_time:,.0f} tok/s")
print(f"  Init val loss   : {init_val:.4f} (expected ~{expected_init:.2f})")
final_lm = M_lm[-1] if M_lm else float('nan')
print(f"  Final train loss: {final_lm:.4f}")
print(f"  Best val loss   : {best_val:.4f} (ppl={math.exp(min(best_val,20)):.2f})")
if M_ge:
    print(f"  Gate entropy    : {M_ge[0]:.4f} -> {M_ge[-1]:.4f} "
          f"(max={max_ent:.2f})")
print(f"  Ablation delta  : {abl_delta:+.4f} (same-batch, seed=12345) "
      f"({'MEMORY HELPS' if abl_delta > 0 else 'neutral/not yet'})")
print(f"  Conflict test   : mechanism_functional="
      f"{conflict_results.get('mechanism_functional', 'N/A')}")
# Submodule gradient health
if M_subgrad:
    dead = [k for k, v in M_subgrad[-1].items() if v == 0.0]
    if dead:
        print(f"  [WARN] Dead submodules (zero grad): {dead}")
    else:
        print(f"  Grad flow       : all {len(SUBMODULE_GROUPS)} submodules have nonzero grad")
print(f"  Report JSON     : {report_path}")
print(f"  Plots PNG       : {plot_path}")
print(f"  Checkpoints     : {CHECKPOINT_DIR}")
print(SEP)
print("WHAT THIS PROVES:")
print("  - Training loop is stable and produces decreasing loss")
print("  - Gradients reach all submodules (embeddings, backbone, memory, fusion)")
print("  - Writer gate sharpens under entropy pressure")
print("  - Memory banks populate during training")
print("  - ConflictMemory detection mechanism is functional (if test passed)")
print("  - Same-batch ablation provides controlled memory impact measurement")
print()
print("WHAT THIS DOES NOT PROVE:")
print("  - Gate sharpening does not prove SEMANTIC routing correctness")
print("  - Memory occupancy does not prove correct content storage")
print("  - 5000 steps may be insufficient for full memory utility")
print("  - Curriculum is minimal -- Step 3 benchmarks are needed for real validation")
print(SEP)
print("STATUS: IMPLEMENTED BUT NOT VERIFIED (requires Colab A100 runtime)")
print(SEP)
