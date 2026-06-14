# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- Step 2 v11 training, adapted for LOCAL single-GPU
# Windows execution on an NVIDIA RTX 5080 (Blackwell, sm_120).
#
# Adapted from colab/step2_training_v11.py. Minimal intervention: the model
# architecture (dcortex/ package) is imported as-is and NOT modified. The
# episode generation, loss formulation, and evaluation logic are preserved.
#
# Local changes only:
#   - No Google Drive mount, no inline write_source; the real dcortex package
#     is imported from the repo root.
#   - Local PROJECT_ROOT under runs/, all file I/O encoding='utf-8'.
#   - bf16 autocast, NO GradScaler (Blackwell takes the A100 path).
#   - Gradient checkpointing on the decoder standard blocks (training-side
#     wrap, no architecture edit).
#   - LM slice fed from a local BYON jsonl corpus tokenized to .bin.
#   - Atomic checkpoints (.tmp then os.rename), full content, numeric-sort
#     resume via int(name.split("_step_")[1].split(".")[0]), try/except for
#     corrupt checkpoints, RNG and loss_history persisted.
#   - OOM guard (del tensors; gc.collect(); torch.cuda.empty_cache()).
#   - Per-step log "Step X/Y | loss | tok/s | ETA", an aggregate train_loss
#     series for the LEARNS gate, peak VRAM tracking for the FIT gate, and a
#     per-run metadata JSON.

import argparse
import contextlib
import gc
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Windows console is cp1252 by default; force utf-8 so status glyphs print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - older streams without reconfigure
        pass

# --- Repo on path so the canonical dcortex package imports cleanly ---------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import StandardTransformerBlock

SEP: str = "=" * 70
SEED: int = 42

DEFAULT_LM_SOURCE: str = r"E:\DATA\training_data_clean\06_FINAL_TRAINING_READY-003_BYON.jsonl"


# ===========================================================================
# SILENT CONTEXT (suppress the model's reset_memory chatter inside the loop)
# ===========================================================================

@contextlib.contextmanager
def silent_stdout():
    """Redirect stdout to a throwaway buffer for noisy model calls."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ===========================================================================
# GRADIENT CHECKPOINTING (decoder standard blocks, training-side only)
# ===========================================================================

_GRAD_CHECKPOINT: Dict[str, bool] = {"enabled": False}
_ORIG_STD_BLOCK_FORWARD = StandardTransformerBlock.forward


def _checkpointed_std_block_forward(
    self: StandardTransformerBlock,
    h: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrap the stateless standard block in activation checkpointing.

    Only active during training with grad enabled on a grad-bearing input.
    use_reentrant=False preserves autocast and RNG. Stateless block: the
    recompute has no side effects on memory banks.
    """
    if (_GRAD_CHECKPOINT["enabled"] and self.training
            and torch.is_grad_enabled() and h.requires_grad):
        return torch_checkpoint.checkpoint(
            _ORIG_STD_BLOCK_FORWARD, self, h, attention_mask,
            use_reentrant=False,
        )
    return _ORIG_STD_BLOCK_FORWARD(self, h, attention_mask)


def enable_gradient_checkpointing() -> None:
    """Install the decoder standard-block checkpointing wrapper."""
    StandardTransformerBlock.forward = _checkpointed_std_block_forward
    _GRAD_CHECKPOINT["enabled"] = True
    print("[INFO] Gradient checkpointing ON (decoder standard blocks)", flush=True)


# ===========================================================================
# ENVIRONMENT / GPU
# ===========================================================================

def setup_device() -> Tuple[torch.device, torch.dtype, bool]:
    """Select device and precision policy. Blackwell -> A100 path (bf16)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Install torch>=2.7 cu128 in the venv and "
            "verify the NVIDIA driver. Run scripts/verify_env_local.py first."
        )
    gpu_name: str = torch.cuda.get_device_name(0)
    gpu_mem_gb: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cap = torch.cuda.get_device_capability(0)

    print(SEP, flush=True)
    print(f"[INFO] GPU: {gpu_name} | VRAM: {gpu_mem_gb:.1f} GB | "
          f"SM {cap[0]}.{cap[1]}", flush=True)

    if "A100" in gpu_name or cap[0] >= 8:
        dtype = torch.bfloat16
        use_scaler = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] Blackwell/A100 mode: bfloat16 autocast, TF32, NO GradScaler",
              flush=True)
    else:
        dtype = torch.float16
        use_scaler = True
        print(f"[WARN] {gpu_name}: fp16 + GradScaler path", flush=True)

    torch.backends.cudnn.benchmark = True
    print(SEP, flush=True)
    return torch.device("cuda"), dtype, use_scaler


def seed_everything(seed: int) -> None:
    """Seed random, numpy, and torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# TOKENIZER + LOCAL LM CORPUS -> .bin
# ===========================================================================

ENC = tiktoken.get_encoding("gpt2")
EOT: int = ENC.eot_token


def tokenize_corpus_to_bin(
    source_path: str,
    bin_dir: str,
    train_tokens: int,
    val_tokens: int,
) -> Tuple[str, str]:
    """Tokenize a local BYON jsonl ({"text": ...}) into train/val .bin files.

    One streaming pass fills train first, then val (disjoint, no overlap).
    uint16 since the gpt2 vocab (50257) fits. Atomic write. Cached on rerun.
    Raises RuntimeError if the source is missing or yields too few tokens.
    """
    train_path = os.path.join(bin_dir, "byon_train.bin")
    val_path = os.path.join(bin_dir, "byon_val.bin")
    if (os.path.exists(train_path) and os.path.getsize(train_path) >= train_tokens * 2
            and os.path.exists(val_path) and os.path.getsize(val_path) >= val_tokens * 2):
        nt = os.path.getsize(train_path) // 2
        nv = os.path.getsize(val_path) // 2
        print(f"[INFO] LM corpus cached: {nt:,} train / {nv:,} val tokens", flush=True)
        return train_path, val_path

    if not os.path.exists(source_path):
        raise RuntimeError(
            f"LM corpus source not found: {source_path}. Provide a valid local "
            f"jsonl with a 'text' field via --lm-source, or point to another "
            f"E:\\DATA\\training_data_clean file. No synthetic fallback is used."
        )

    need_total = train_tokens + val_tokens
    print(f"[INFO] Tokenizing {os.path.basename(source_path)} -> "
          f"{need_total:,} tokens (gpt2)...", flush=True)

    tokens: List[int] = []
    lines_read = 0
    with open(source_path, "r", encoding="utf-8") as handle:
        for line in handle:
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "") or obj.get("story", "")
            if not text:
                continue
            tokens.extend(ENC.encode_ordinary(text))
            tokens.append(EOT)
            if len(tokens) >= need_total:
                break
            if lines_read % 20000 == 0:
                print(f"  {len(tokens):,} tok ({lines_read:,} lines)", flush=True)

    if len(tokens) < need_total:
        raise RuntimeError(
            f"LM corpus produced only {len(tokens):,} tokens (< {need_total:,} "
            f"required) from {source_path}. Provide a larger source or lower "
            f"--lm-train-tokens / --lm-val-tokens."
        )

    train_arr = np.array(tokens[:train_tokens], dtype=np.uint16)
    val_arr = np.array(tokens[train_tokens:train_tokens + val_tokens], dtype=np.uint16)
    for arr, path in [(train_arr, train_path), (val_arr, val_path)]:
        tmp = path + ".tmp"
        arr.tofile(tmp)
        os.rename(tmp, path)
    print(f"[INFO] LM corpus: {len(train_arr):,} train / {len(val_arr):,} val tokens",
          flush=True)
    return train_path, val_path


# ===========================================================================
# TRAIN CONFIG
# ===========================================================================

@dataclass
class TrainConfig:
    seq_len: int = 64
    lm_batch: int = 8
    grad_accum: int = 16
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    total_steps: int = 300
    warmup_steps: int = 30
    min_facts: int = 3
    max_facts: int = 5
    # Loss weights
    w_emit: float = 1.0
    w_sel: float = 1.0
    w_sep_neg: float = 0.5
    # Episode mix (structural / lm)
    lm_ratio: float = 0.25
    # Within structural: simple / update / distractor
    simple_ratio: float = 0.5
    update_ratio: float = 0.25
    distractor_ratio: float = 0.25
    # Lexical binding
    lexical_alpha: float = 0.9
    # LR multipliers per param group
    addr_lr_mult: float = 3.0
    enc_lr_mult: float = 2.0


# ===========================================================================
# EPISODE GENERATOR (ported verbatim from v11, behavior preserved)
# ===========================================================================

_ENTITIES = ["cat", "dog", "bird", "fish", "rabbit", "horse", "bear", "fox",
             "lion", "tiger", "monkey", "penguin", "owl", "wolf", "deer",
             "dragon", "knight", "wizard", "princess", "fairy", "goblin", "witch",
             "pirate", "giant", "ghost", "robot", "queen", "king", "dwarf", "elf"]
_COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink",
           "orange", "purple", "golden", "silver", "crimson", "gray", "violet"]
_CLUSTER_ANIMALS = ["cat", "dog", "bird", "fish", "rabbit", "horse", "bear", "fox",
                    "lion", "tiger", "monkey", "penguin", "owl", "wolf", "deer"]
_CLUSTER_FANTASY = ["dragon", "knight", "wizard", "princess", "fairy", "goblin",
                    "witch", "pirate", "giant", "ghost", "robot", "queen", "king",
                    "dwarf", "elf"]


@dataclass
class FactInfo:
    text: str
    entity: str
    value: str
    fact_idx: int
    answer_token_id: int


@dataclass
class EpisodeGT:
    facts: List[FactInfo]
    prompt: str
    target_fact_idx: int
    answer_token_id: int
    ep_type: str


def generate_simple_episode(tc: TrainConfig, n_facts: int = 0) -> EpisodeGT:
    if n_facts == 0:
        n_facts = random.randint(tc.min_facts, tc.max_facts)
    entities = random.sample(_ENTITIES, n_facts)
    colors = random.sample(_COLORS, n_facts)
    target = random.randint(0, n_facts - 1)
    facts: List[FactInfo] = []
    for i, (e, c) in enumerate(zip(entities, colors)):
        ans_tok = ENC.encode_ordinary(f" {c}")[0]
        facts.append(FactInfo(f"The {e} is {c}.", e, c, i, ans_tok))
    prompt = f"What color is the {entities[target]}? The {entities[target]} is"
    return EpisodeGT(facts, prompt, target, facts[target].answer_token_id, "simple")


def generate_update_episode(tc: TrainConfig, n_facts: int = 0) -> EpisodeGT:
    if n_facts == 0:
        n_facts = random.randint(tc.min_facts, tc.max_facts)
    entities = random.sample(_ENTITIES, n_facts)
    colors = random.sample(_COLORS, n_facts + 1)
    update_target = random.randint(0, n_facts - 1)
    new_color = colors[n_facts]
    facts: List[FactInfo] = []
    for i, (e, c) in enumerate(zip(entities, colors[:n_facts])):
        ans_tok = ENC.encode_ordinary(f" {c}")[0]
        facts.append(FactInfo(f"The {e} is {c}.", e, c, i, ans_tok))
    upd_ans_tok = ENC.encode_ordinary(f" {new_color}")[0]
    facts.append(FactInfo(
        f"The {entities[update_target]} is now {new_color}.",
        entities[update_target], new_color, n_facts, upd_ans_tok,
    ))
    prompt = f"What color is the {entities[update_target]} now? The {entities[update_target]} is"
    return EpisodeGT(facts, prompt, n_facts, upd_ans_tok, "update")


def generate_distractor_episode(tc: TrainConfig, n_facts: int = 0) -> EpisodeGT:
    if n_facts == 0:
        n_facts = random.randint(tc.min_facts, tc.max_facts)
    cluster = random.choice([_CLUSTER_ANIMALS, _CLUSTER_FANTASY])
    n_facts = min(n_facts, len(cluster))
    entities = random.sample(cluster, n_facts)
    colors = random.sample(_COLORS, n_facts)
    target = random.randint(0, n_facts - 1)
    facts: List[FactInfo] = []
    for i, (e, c) in enumerate(zip(entities, colors)):
        ans_tok = ENC.encode_ordinary(f" {c}")[0]
        facts.append(FactInfo(f"The {e} is {c}.", e, c, i, ans_tok))
    prompt = f"What color is the {entities[target]}? The {entities[target]} is"
    return EpisodeGT(facts, prompt, target, facts[target].answer_token_id, "distractor")


def generate_episode(tc: TrainConfig) -> EpisodeGT:
    r = random.random()
    if r < tc.simple_ratio:
        return generate_simple_episode(tc)
    if r < tc.simple_ratio + tc.update_ratio:
        return generate_update_episode(tc)
    return generate_distractor_episode(tc)


def _pad(ids: List[int], length: int) -> List[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [EOT] * (length - len(ids))


def encode_text(text: str) -> List[int]:
    return ENC.encode_ordinary(text)


# ===========================================================================
# STRUCTURAL EPISODE LOSS (ported from v11, behavior preserved)
# ===========================================================================

def run_structural_episode(
    model: DCortexV2Model,
    ep: EpisodeGT,
    tc: TrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], int]:
    """Run one structural episode. Returns (loss, metrics, tokens_processed)."""
    with silent_stdout():
        model.reset_memory()
    model.begin_episode()

    tokens_processed = 0
    fact_keys: List[torch.Tensor] = []
    for fact in ep.facts:
        f_ids = _pad(encode_text(fact.text) + [EOT], tc.seq_len)
        xf = torch.tensor([f_ids], dtype=torch.long, device=device)
        tokens_processed += xf.numel()
        ans_id = torch.tensor([fact.answer_token_id], dtype=torch.long, device=device)
        aux = model.encode(xf, answer_token_id=ans_id,
                           lexical_alpha=tc.lexical_alpha, force_bank="working")
        fact_keys.append(aux["w_k_ent"][0])

    p_ids = encode_text(ep.prompt)
    xp = torch.tensor([_pad(p_ids, tc.seq_len)], dtype=torch.long, device=device)
    tokens_processed += xp.numel()
    _, retrieved = model.decode(xp, return_retrieved=True)
    aux_logits = model.aux_answer_head(retrieved)

    target_tensor = torch.tensor([ep.answer_token_id], device=device)
    l_emit = F.cross_entropy(aux_logits, target_tensor)

    # L_sel
    b, t = xp.shape
    pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    q_emb = model.shared_token_emb(xp) + model.shared_pos_emb(pos)
    q_addr = model.shared_address_encoder(q_emb)
    q_k_ent, _, _ = model.shared_query_engine(q_addr)
    keys = F.normalize(torch.stack(fact_keys, dim=0), dim=-1)
    q_n = F.normalize(q_k_ent, dim=-1)
    sim = (q_n @ keys.t()).squeeze(0)
    log_p = F.log_softmax(sim * 5.0, dim=-1)
    l_sel = -log_p[ep.target_fact_idx]

    # L_sep_neg
    l_sep_neg = torch.tensor(0.0, device=device)
    if len(fact_keys) >= 2:
        keys_n = F.normalize(torch.stack(fact_keys, dim=0), dim=-1)
        sims_off = keys_n @ keys_n.t()
        mask = torch.eye(len(fact_keys), device=device, dtype=torch.bool)
        if ep.ep_type == "update":
            n_f = len(ep.facts)
            upd_orig = None
            for i, f in enumerate(ep.facts[:-1]):
                if f.entity == ep.facts[-1].entity:
                    upd_orig = i
                    break
            if upd_orig is not None:
                mask[upd_orig, n_f - 1] = True
                mask[n_f - 1, upd_orig] = True
        off_diag = sims_off[~mask]
        l_sep_neg = F.relu(off_diag - 0.5).pow(2).mean()

    total = tc.w_emit * l_emit + tc.w_sel * l_sel + tc.w_sep_neg * l_sep_neg

    with torch.no_grad():
        pred = aux_logits[0].argmax().item()
        top1 = pred == ep.answer_token_id

    metrics = {
        "L_emit": l_emit.item(),
        "L_sel": l_sel.item(),
        "L_sep_neg": l_sep_neg.item() if isinstance(l_sep_neg, torch.Tensor) else l_sep_neg,
        "top1": float(top1),
        "ep_type": ep.ep_type,
    }
    return total, metrics, tokens_processed


# ===========================================================================
# EVAL (ported from v11)
# ===========================================================================

@torch.no_grad()
def eval_type(model: DCortexV2Model, ep_gen_fn, tc: TrainConfig,
              device: torch.device, dtype: torch.dtype, n: int = 200) -> float:
    model.eval()
    correct = 0
    for _ in range(n):
        ep = ep_gen_fn(tc)
        with silent_stdout():
            model.reset_memory()
        for fact in ep.facts:
            f_ids = _pad(encode_text(fact.text) + [EOT], tc.seq_len)
            xf = torch.tensor([f_ids], dtype=torch.long, device=device)
            ans_id = torch.tensor([fact.answer_token_id], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=dtype):
                model.encode(xf, answer_token_id=ans_id,
                             lexical_alpha=tc.lexical_alpha, force_bank="working")
        p_ids = encode_text(ep.prompt)
        xp = torch.tensor([_pad(p_ids, tc.seq_len)], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            _, retrieved = model.decode(xp, return_retrieved=True)
            aux_logits = model.aux_answer_head(retrieved).float()
        if aux_logits[0].argmax().item() == ep.answer_token_id:
            correct += 1
    model.train()
    return correct / n


@torch.no_grad()
def eval_lm(model: DCortexV2Model, val_data: np.memmap, tc: TrainConfig,
            device: torch.device, dtype: torch.dtype, vocab_size: int) -> float:
    model.eval()
    total = 0.0
    n = 20
    for _ in range(n):
        ix = np.random.randint(0, len(val_data) - tc.seq_len - 1, size=(tc.lm_batch,))
        x = np.stack([val_data[i:i + tc.seq_len].astype(np.int64) for i in ix])
        y = np.stack([val_data[i + 1:i + 1 + tc.seq_len].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with silent_stdout():
            model.reset_memory()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model.decode(x)
            total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return total / n


# ===========================================================================
# CHECKPOINTING (atomic, full content, numeric-sort resume)
# ===========================================================================

def save_ckpt(ckpt_dir: str, model: DCortexV2Model, optimizer: torch.optim.Optimizer,
              step: int, epoch: int, loss_history: List[Tuple[int, float]],
              model_cfg: DCortexConfig, train_cfg: TrainConfig, best_val: float) -> str:
    """Atomic checkpoint write (.tmp then os.rename). Full content."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss_history": loss_history,
        "best_val": best_val,
        "config_model": asdict(model_cfg),
        "config_train": asdict(train_cfg),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    fname = f"ckpt_dcortex_step_{step:06d}.pt"
    final = os.path.join(ckpt_dir, fname)
    tmp = final + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, final)  # atomic on Windows and POSIX (rename raises if dst exists on Windows)
    print(f"[INFO] Checkpoint: {fname}", flush=True)
    return final


def load_latest_ckpt(ckpt_dir: str, model: DCortexV2Model,
                     optimizer: torch.optim.Optimizer, device: torch.device
                     ) -> Tuple[int, int, List[Tuple[int, float]], float]:
    """Resume from the highest-step valid checkpoint. Numeric sort.

    Returns (start_step, epoch, loss_history, best_val). Fresh -> (0, 0, [], inf).
    Corrupt checkpoints are skipped (try/except), falling back to earlier ones.
    """
    paths = list(Path(ckpt_dir).glob("ckpt_dcortex_step_*.pt"))
    if not paths:
        return 0, 0, [], float("inf")
    paths.sort(key=lambda p: int(p.name.split("_step_")[1].split(".")[0]))
    for path in reversed(paths):
        try:
            c = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(c["model"])
            optimizer.load_state_dict(c["optimizer"])
            rng = c.get("rng")
            if rng is not None:
                try:
                    random.setstate(rng["python"])
                    np.random.set_state(rng["numpy"])
                    torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
                    torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in rng["cuda"]])
                except Exception as rng_exc:  # noqa: BLE001
                    print(f"[WARN] RNG restore skipped: {rng_exc}", flush=True)
            print(f"[INFO] Resumed from {path.name} (step {c['step']})", flush=True)
            return (int(c["step"]), int(c.get("epoch", 0)),
                    list(c.get("loss_history", [])), float(c.get("best_val", float("inf"))))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Corrupt checkpoint {path.name} skipped: {exc}", flush=True)
            continue
    return 0, 0, [], float("inf")


# ===========================================================================
# OPTIMIZER (param groups ported from v11)
# ===========================================================================

def build_optimizer(model: DCortexV2Model, tc: TrainConfig
                    ) -> Tuple[torch.optim.Optimizer, List[float]]:
    shared_ids = set()
    shared_d, shared_nd = [], []
    for n, p in model.named_parameters():
        if (n.startswith("shared_") or n.startswith("aux_answer_head")
                or n.startswith("value_to_key_proj")):
            shared_ids.add(id(p))
            (shared_nd if p.dim() < 2 or "norm" in n or "bias" in n else shared_d).append(p)

    enc_d, enc_nd = [], []
    for n, p in model.encoder.named_parameters():
        if id(p) in shared_ids or not p.requires_grad:
            continue
        (enc_nd if p.dim() < 2 or "norm" in n or "bias" in n else enc_d).append(p)

    dec_d, dec_nd = [], []
    for n, p in model.named_parameters():
        if n.startswith("dec_") and id(p) not in shared_ids and p.requires_grad:
            (dec_nd if p.dim() < 2 or "norm" in n or "bias" in n else dec_d).append(p)

    optimizer = torch.optim.AdamW([
        {"params": shared_d, "weight_decay": tc.weight_decay, "lr": tc.lr * tc.addr_lr_mult},
        {"params": shared_nd, "weight_decay": 0.0, "lr": tc.lr * tc.addr_lr_mult},
        {"params": enc_d, "weight_decay": tc.weight_decay, "lr": tc.lr * tc.enc_lr_mult},
        {"params": enc_nd, "weight_decay": 0.0, "lr": tc.lr * tc.enc_lr_mult},
        {"params": dec_d, "weight_decay": tc.weight_decay, "lr": tc.lr},
        {"params": dec_nd, "weight_decay": 0.0, "lr": tc.lr},
    ], lr=tc.lr, betas=(0.9, 0.95))

    lr_mults = [tc.addr_lr_mult, tc.addr_lr_mult, tc.enc_lr_mult, tc.enc_lr_mult, 1.0, 1.0]
    return optimizer, lr_mults


def get_lr(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= tc.total_steps:
        return tc.min_lr
    t = (step - tc.warmup_steps) / max(1, tc.total_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * t))


# ===========================================================================
# TRAINING
# ===========================================================================

def train(args: argparse.Namespace) -> Dict[str, object]:
    device, dtype, use_scaler = setup_device()
    if use_scaler:
        raise RuntimeError(
            "fp16+GradScaler path selected, but this bring-up mandates bf16/"
            "no-GradScaler. Expected a compute-capability >= 8 GPU (RTX 5080)."
        )
    seed_everything(SEED)
    enable_gradient_checkpointing()

    project_root = Path(args.run_dir)
    ckpt_dir = project_root / "checkpoints"
    results_dir = project_root / "results"
    logs_dir = project_root / "logs"
    bin_dir = project_root / "dataset_cache" / "bin"
    configs_dir = project_root / "configs"
    for d in [ckpt_dir, results_dir, logs_dir, bin_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Project root: {project_root}", flush=True)

    tc = TrainConfig(total_steps=args.total_steps, warmup_steps=args.warmup_steps)
    with open(configs_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(tc), handle, indent=2)

    # --- LM corpus ---
    train_bin, val_bin = tokenize_corpus_to_bin(
        args.lm_source, str(bin_dir), args.lm_train_tokens, args.lm_val_tokens)
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    print(f"[INFO] LM data: {len(train_data):,} train / {len(val_data):,} val tokens",
          flush=True)

    # --- Model + optimizer ---
    model_cfg = DCortexConfig()
    model = DCortexV2Model(model_cfg).to(device)
    optimizer, lr_mults = build_optimizer(model, tc)

    start_step, epoch, loss_history, best_val = 0, 0, [], float("inf")
    if not args.fresh:
        start_step, epoch, loss_history, best_val = load_latest_ckpt(
            ckpt_dir=str(ckpt_dir), model=model, optimizer=optimizer, device=device)
    if start_step == 0:
        print("[INFO] Starting fresh.", flush=True)

    max_steps = args.max_steps if args.max_steps > 0 else tc.total_steps
    vocab_size = model_cfg.vocab_size

    print(SEP, flush=True)
    print(f"[INFO] D_CORTEX v2.0-alpha LOCAL training | tag={args.tag}", flush=True)
    print(f"  steps {start_step} -> {max_steps} (LR horizon {tc.total_steps}) | "
          f"accum {tc.grad_accum} | seq_len {tc.seq_len} | lm_ratio {tc.lm_ratio}",
          flush=True)
    print(SEP, flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    peak_vram_at_50 = 0.0
    oom_steps = 0
    last_saved_step = -1
    resumed_first_loss: Optional[float] = None

    for step in range(start_step, max_steps):
        lr = get_lr(step, tc)
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = lr * lr_mults[i]
        optimizer.zero_grad(set_to_none=True)

        step_t0 = time.time()
        step_loss = 0.0
        step_tokens = 0
        emit_acc = 0.0
        n_lm = 0
        try:
            for _ in range(tc.grad_accum):
                if random.random() < tc.lm_ratio:
                    ix = np.random.randint(0, len(train_data) - tc.seq_len - 1,
                                           size=(tc.lm_batch,))
                    x = np.stack([train_data[i:i + tc.seq_len].astype(np.int64) for i in ix])
                    y = np.stack([train_data[i + 1:i + 1 + tc.seq_len].astype(np.int64) for i in ix])
                    x = torch.from_numpy(x).to(device)
                    y = torch.from_numpy(y).to(device)
                    with silent_stdout():
                        model.reset_memory()
                    with torch.amp.autocast("cuda", dtype=dtype):
                        logits = model.decode(x)
                        lm_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                        scaled = lm_loss / tc.grad_accum
                    scaled.backward()
                    step_loss += lm_loss.item() / tc.grad_accum
                    step_tokens += int(x.numel())
                    n_lm += 1
                else:
                    ep = generate_episode(tc)
                    with torch.amp.autocast("cuda", dtype=dtype):
                        total, m, ep_tokens = run_structural_episode(model, ep, tc, device)
                        scaled = total / tc.grad_accum
                    scaled.backward()
                    model.clear_overlays()
                    step_loss += total.item() / tc.grad_accum
                    step_tokens += ep_tokens
                    emit_acc += m["L_emit"] / tc.grad_accum
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                oom_steps += 1
                print(f"[WARN] OOM at step {step}, recovering and skipping.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                for _name in ["x", "y", "logits", "lm_loss", "scaled", "total"]:
                    if _name in dir():
                        pass
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        optimizer.step()

        loss_history.append((step, step_loss))
        if resumed_first_loss is None and start_step > 0:
            resumed_first_loss = step_loss

        step_dt = max(1e-6, time.time() - step_t0)
        tok_s = step_tokens / step_dt
        peak_now = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if step >= 50 and peak_vram_at_50 == 0.0:
            peak_vram_at_50 = peak_now

        if step % args.log_every == 0 or step == start_step or step == max_steps - 1:
            elapsed = time.time() - t_start
            done = step - start_step + 1
            eta_s = elapsed / max(1, done) * (max_steps - step - 1)
            print(f"Step {step:5d}/{max_steps} | loss {step_loss:.4f} | "
                  f"emit {emit_acc:.3f} | tok/s {tok_s:8.0f} | "
                  f"gn {grad_norm:.2f} | peakVRAM {peak_now:.2f}GB | "
                  f"ETA {int(eta_s // 60)}m{int(eta_s % 60):02d}s", flush=True)

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(str(ckpt_dir), model, optimizer, step + 1, epoch,
                      loss_history, model_cfg, tc, best_val)
            last_saved_step = step + 1

        if math.isnan(step_loss) or math.isinf(step_loss):
            print(f"[ERROR] Non-finite loss at step {step}. Stopping.", flush=True)
            break

    total_time = time.time() - t_start
    final_step = loss_history[-1][0] + 1 if loss_history else start_step
    # Save a final checkpoint at the stop point unless it was already saved by
    # the periodic checkpointer this segment.
    if final_step != last_saved_step:
        save_ckpt(str(ckpt_dir), model, optimizer, final_step, epoch,
                  loss_history, model_cfg, tc, best_val)

    peak_vram_overall = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    print(SEP, flush=True)
    print(f"[INFO] Segment complete in {total_time / 60:.2f} min | "
          f"peak alloc {peak_vram_overall:.2f} GB | peak reserved {peak_reserved:.2f} GB | "
          f"OOM skips {oom_steps}", flush=True)
    print(SEP, flush=True)

    # --- Persist artifacts ---
    losses = [v for (_s, v) in loss_history]
    history_path = results_dir / "loss_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({"loss_history": loss_history}, handle, indent=2)

    meta = {
        "tag": args.tag,
        "start_step": start_step,
        "end_step": final_step,
        "max_steps": max_steps,
        "total_steps_horizon": tc.total_steps,
        "n_logged": len(loss_history),
        "peak_vram_alloc_gb": round(peak_vram_overall, 3),
        "peak_vram_reserved_gb": round(peak_reserved, 3),
        "peak_vram_at_step50_gb": round(peak_vram_at_50, 3),
        "oom_steps": oom_steps,
        "minutes": round(total_time / 60, 3),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "resumed": start_step > 0,
        "resumed_first_loss": resumed_first_loss,
        "gradient_checkpointing": _GRAD_CHECKPOINT["enabled"],
        "dtype": str(dtype),
    }
    meta_path = results_dir / f"run_meta_{args.tag}.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[INFO] Wrote {meta_path.name} and {history_path.name}", flush=True)
    print("RUN_META_JSON " + json.dumps(meta), flush=True)
    return meta


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D_Cortex local single-GPU training")
    default_run_dir = str(REPO_ROOT / "runs" / "dcortex")
    p.add_argument("--run-dir", type=str, default=default_run_dir)
    p.add_argument("--total-steps", type=int, default=300, help="LR schedule horizon")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=0, help="Stop at this step (0=total)")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--lm-source", type=str, default=DEFAULT_LM_SOURCE)
    p.add_argument("--lm-train-tokens", type=int, default=5_000_000)
    p.add_argument("--lm-val-tokens", type=int, default=500_000)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
