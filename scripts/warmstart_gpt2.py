# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- warm-start the decoder from pretrained gpt2-medium.
# Produces warmstarted_init.pt: a DCortexV2Model (campaign BIG config) whose
# standard decoder blocks, the self-attn/FFN of the fusion blocks, token and
# positional embeddings, and final norm are initialized from gpt2-medium, with
# the fusion memory cross-attention zero-initialized so it is inert at init.
# The dcortex/ architecture is NOT modified; this is weight initialization only.

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch
import torch.nn as nn

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import StandardTransformerBlock
from dcortex.backbone.fusion_block import FusionBlock

SEP: str = "=" * 70


def big_config() -> DCortexConfig:
    return DCortexConfig(
        hidden_dim=1024, n_enc_heads=16, n_dec_heads=16,
        enc_ff_dim=4096, dec_ff_dim=4096, n_dec_layers=16,
        n_enc_layers=4, n_fusion_layers=4, max_seq_len=2048,
    )


def _copy(dst: torch.Tensor, src: torch.Tensor, transpose: bool, name: str,
          mapped: List[str]) -> None:
    """Copy src into dst (optionally transposed), asserting shape match."""
    source = src.t().contiguous() if transpose else src
    if tuple(dst.shape) != tuple(source.shape):
        raise ValueError(f"shape mismatch for {name}: dst {tuple(dst.shape)} "
                         f"vs src {tuple(source.shape)} (transpose={transpose})")
    with torch.no_grad():
        dst.copy_(source)
    mapped.append(name)


def map_block(d_block: nn.Module, g_block: nn.Module, prefix: str,
              mapped: List[str]) -> None:
    """Map one GPT-2 block into a StandardTransformerBlock or a FusionBlock's
    self-attn/FFN. norm1/norm_self, attn/self_attn, norm2/norm_ff, ff."""
    is_fusion = isinstance(d_block, FusionBlock)
    norm_a = d_block.norm_self if is_fusion else d_block.norm1
    attn = d_block.self_attn if is_fusion else d_block.attn
    norm_b = d_block.norm_ff if is_fusion else d_block.norm2

    _copy(norm_a.weight, g_block.ln_1.weight, False, f"{prefix}.norm1/self", mapped)
    _copy(norm_a.bias, g_block.ln_1.bias, False, f"{prefix}.norm1/self.bias", mapped)
    # Conv1D weights are [in, out]; transpose into Linear [out, in].
    _copy(attn.qkv.weight, g_block.attn.c_attn.weight, True, f"{prefix}.attn.qkv.weight", mapped)
    _copy(attn.qkv.bias, g_block.attn.c_attn.bias, False, f"{prefix}.attn.qkv.bias", mapped)
    _copy(attn.out.weight, g_block.attn.c_proj.weight, True, f"{prefix}.attn.out.weight", mapped)
    _copy(attn.out.bias, g_block.attn.c_proj.bias, False, f"{prefix}.attn.out.bias", mapped)
    _copy(norm_b.weight, g_block.ln_2.weight, False, f"{prefix}.norm2/ff", mapped)
    _copy(norm_b.bias, g_block.ln_2.bias, False, f"{prefix}.norm2/ff.bias", mapped)
    _copy(d_block.ff.fc1.weight, g_block.mlp.c_fc.weight, True, f"{prefix}.ff.fc1.weight", mapped)
    _copy(d_block.ff.fc1.bias, g_block.mlp.c_fc.bias, False, f"{prefix}.ff.fc1.bias", mapped)
    _copy(d_block.ff.fc2.weight, g_block.mlp.c_proj.weight, True, f"{prefix}.ff.fc2.weight", mapped)
    _copy(d_block.ff.fc2.bias, g_block.mlp.c_proj.bias, False, f"{prefix}.ff.fc2.bias", mapped)


def warmstart(out_dir: str) -> Dict:
    print(SEP, flush=True)
    print("[HF] Loading gpt2-medium ...", flush=True)
    from transformers import GPT2LMHeadModel
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    gpt2.eval()
    gconf = gpt2.config
    assert (gconf.n_embd, gconf.n_head, gconf.vocab_size) == (1024, 16, 50257), \
        "gpt2-medium dims unexpected"
    print(f"[HF] gpt2-medium: {gconf.n_layer} layers, n_embd {gconf.n_embd}, "
          f"heads {gconf.n_head}, vocab {gconf.vocab_size}, act {gconf.activation_function}",
          flush=True)

    cfg = big_config()
    model = DCortexV2Model(cfg)
    model.eval()

    mapped: List[str] = []
    zeroed: List[str] = []

    # --- Embeddings + final norm ---
    _copy(model.shared_token_emb.weight, gpt2.transformer.wte.weight, False,
          "shared_token_emb.weight", mapped)
    with torch.no_grad():
        model.shared_pos_emb.weight[:1024].copy_(gpt2.transformer.wpe.weight)
    mapped.append("shared_pos_emb.weight[:1024]")
    _copy(model.dec_final_norm.weight, gpt2.transformer.ln_f.weight, False,
          "dec_final_norm.weight", mapped)
    _copy(model.dec_final_norm.bias, gpt2.transformer.ln_f.bias, False,
          "dec_final_norm.bias", mapped)

    # --- 16 blocks: 12 standard + 4 fusion self-attn/FFN ---
    indices = [round(i * 24 / 16) for i in range(16)]
    print(f"[INFO] GPT-2 layer indices selected: {indices}", flush=True)
    dec_blocks = list(model.dec_standard_blocks) + list(model.dec_fusion_blocks)
    assert len(dec_blocks) == 16
    for d_idx, (d_block, g_idx) in enumerate(zip(dec_blocks, indices)):
        kind = "fusion" if isinstance(d_block, FusionBlock) else "std"
        map_block(d_block, gpt2.transformer.h[g_idx], f"dec[{d_idx}:{kind}]<-g{g_idx}", mapped)

    # --- Zero-init fusion memory cross-attention output (make memory inert) ---
    for f_idx, fblock in enumerate(model.dec_fusion_blocks):
        with torch.no_grad():
            fblock.cross_attn.out.weight.zero_()
            fblock.cross_attn.out.bias.zero_()
        zeroed.append(f"dec_fusion_blocks[{f_idx}].cross_attn.out.weight")
        zeroed.append(f"dec_fusion_blocks[{f_idx}].cross_attn.out.bias")

    # --- Scale-match dec_emb_norm to the gpt2 embedding scale ---
    # GPT-2 feeds RAW wte+wpe (per-token std ~0.12) into block 0. dec_emb_norm
    # (an extra LayerNorm with no GPT-2 source) at weight=1 would normalize each
    # token vector to unit std, inflating the residual-stream scale ~8x and
    # wrecking the warm-start. Set its weight to the measured gpt2 embedding
    # per-token std (bias 0) so dec_emb_norm(emb) ~= emb (scale-preserving).
    with torch.no_grad():
        n_s = 4096
        toks = torch.randint(0, 50257, (n_s,))
        pos = torch.arange(n_s) % 1024
        emb_sample = gpt2.transformer.wte.weight[toks] + gpt2.transformer.wpe.weight[pos]
        emb_std = float(emb_sample.std(dim=1).mean().item())
        model.dec_emb_norm.weight.fill_(emb_std)
        model.dec_emb_norm.bias.zero_()
    mapped.append(f"dec_emb_norm.weight<-gpt2_emb_std({emb_std:.4f})")
    mapped.append("dec_emb_norm.bias<-0")
    print(f"[INFO] dec_emb_norm scale-matched to gpt2 emb std={emb_std:.4f} "
          f"(default weight=1 would inflate residual ~{1.0/emb_std:.1f}x)", flush=True)

    # Tie check (lm_head shares shared_token_emb; setting emb already updated it).
    tied = model.dec_lm_head.weight.data_ptr() == model.shared_token_emb.weight.data_ptr()
    print(f"[INFO] lm_head tied to token embedding: {tied}", flush=True)

    # --- Forward sanity on a real held-out batch ---
    val_bin = REPO_ROOT / "runs" / "campaign" / "dataset_cache" / "bin" / "campaign_val.bin"
    logits_stats: Dict = {}
    if val_bin.exists():
        val = np.memmap(val_bin, dtype=np.uint16, mode="r")
        x = np.stack([val[0:1024].astype(np.int64), val[2048:3072].astype(np.int64)])
        xt = torch.from_numpy(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        xt = xt.to(device)
        import contextlib
        import io
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            model.reset_memory()
            logits = model.decode(xt)
        finite = bool(torch.isfinite(logits).all().item())
        logits_stats = {"shape": list(logits.shape), "finite": finite,
                        "min": float(logits.min()), "max": float(logits.max()),
                        "mean": float(logits.mean()), "std": float(logits.std())}
        print(f"[INFO] forward sanity: logits {logits_stats['shape']} finite={finite} "
              f"min={logits_stats['min']:.3f} max={logits_stats['max']:.3f} "
              f"mean={logits_stats['mean']:.3f} std={logits_stats['std']:.3f}", flush=True)
        if not finite:
            raise RuntimeError("Forward produced non-finite logits; warm-start invalid.")
        model.to("cpu")
    else:
        print(f"[WARN] held-out bin not found at {val_bin}; skipping forward sanity.",
              flush=True)

    # --- Manifest (mapped vs fresh vs zeroed) over the full state_dict ---
    all_keys = set(model.state_dict().keys())
    # Mark which actual state_dict tensors were touched by mapping/zeroing.
    touched_sd: set = set()
    for name, _ in model.named_parameters():
        pass
    # Reconstruct touched state_dict keys precisely:
    touched_sd.add("shared_token_emb.weight")
    touched_sd.add("dec_lm_head.weight")  # tied -> same storage
    touched_sd.add("shared_pos_emb.weight")
    touched_sd.update(["dec_final_norm.weight", "dec_final_norm.bias"])
    touched_sd.update(["dec_emb_norm.weight", "dec_emb_norm.bias"])
    for d_idx, d_block in enumerate(model.dec_standard_blocks):
        base = f"dec_standard_blocks.{d_idx}"
        touched_sd.update([f"{base}.norm1.weight", f"{base}.norm1.bias",
                           f"{base}.attn.qkv.weight", f"{base}.attn.qkv.bias",
                           f"{base}.attn.out.weight", f"{base}.attn.out.bias",
                           f"{base}.norm2.weight", f"{base}.norm2.bias",
                           f"{base}.ff.fc1.weight", f"{base}.ff.fc1.bias",
                           f"{base}.ff.fc2.weight", f"{base}.ff.fc2.bias"])
    for f_idx, f_block in enumerate(model.dec_fusion_blocks):
        base = f"dec_fusion_blocks.{f_idx}"
        touched_sd.update([f"{base}.norm_self.weight", f"{base}.norm_self.bias",
                           f"{base}.self_attn.qkv.weight", f"{base}.self_attn.qkv.bias",
                           f"{base}.self_attn.out.weight", f"{base}.self_attn.out.bias",
                           f"{base}.norm_ff.weight", f"{base}.norm_ff.bias",
                           f"{base}.ff.fc1.weight", f"{base}.ff.fc1.bias",
                           f"{base}.ff.fc2.weight", f"{base}.ff.fc2.bias",
                           f"{base}.cross_attn.out.weight", f"{base}.cross_attn.out.bias"])
    fresh_keys = sorted(all_keys - touched_sd)

    manifest = {
        "source": "gpt2-medium",
        "gpt2_layer_indices": indices,
        "activation_drift": "gpt2 gelu_new(tanh) vs dcortex nn.GELU(exact)",
        "n_mapped_ops": len(mapped),
        "n_zeroed_ops": len(zeroed),
        "mapped_ops": mapped,
        "zeroed_ops": zeroed,
        "n_standard_blocks_mapped": len(model.dec_standard_blocks),
        "n_fusion_blocks_mapped": len(model.dec_fusion_blocks),
        "token_emb_mapped": "shared_token_emb.weight" in touched_sd,
        "final_norm_mapped": "dec_final_norm.weight" in touched_sd,
        "lm_head_tied": tied,
        "pos_emb_fresh_rows": "1024..2047",
        "fresh_state_dict_keys_count": len(fresh_keys),
        "fresh_state_dict_keys": fresh_keys,
        "logits_sanity": logits_stats,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "warmstarted_init.pt")
    tmp = out_path + ".tmp"
    torch.save({"model": model.state_dict(), "config_model": cfg.__dict__,
                "manifest": manifest, "source": "gpt2-medium"}, tmp)
    os.replace(tmp, out_path)
    manifest_path = os.path.join(out_dir, "warmstart_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(SEP, flush=True)
    print(f"✓ Mapped {len(mapped)} tensor ops; zeroed {len(zeroed)}; "
          f"{len(fresh_keys)} fresh state_dict tensors.", flush=True)
    print(f"[INFO] Saved {out_path}", flush=True)
    print(f"[INFO] Saved {manifest_path}", flush=True)
    print(SEP, flush=True)
    return manifest


def main() -> int:
    out_dir = str(REPO_ROOT / "runs" / "warmstart")
    warmstart(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
