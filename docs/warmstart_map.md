# GPT-2-medium -> D_Cortex decoder warm-start map (PHASE A)

Field-by-field mapping used by `scripts/warmstart_gpt2.py`. Verified empirically
against `transformers` GPT2LMHeadModel("gpt2-medium") and a fresh
`DCortexV2Model` at the campaign BIG config. The `dcortex/` architecture is NOT
modified; this is weight initialization only.

## Dimensional match (why gpt2-medium, not gpt2)

| | gpt2-medium | D_Cortex BIG | match |
|---|---|---|---|
| hidden (n_embd) | 1024 | 1024 | yes |
| heads | 16 | 16 (head_dim 64) | yes |
| ff (n_inner) | 4096 | 4096 | yes |
| vocab | 50257 | 50257 | yes |
| positions | 1024 | 2048 | partial (map first 1024) |
| transformer layers | 24 | 16 decoder blocks (12 std + 4 fusion) | subset 16 of 24 |

## Block structure (read, not assumed)

- D_Cortex `StandardTransformerBlock` is PRE-NORM, combined-QKV `nn.Linear`,
  `nn.GELU()`, biases on every Linear, `nn.LayerNorm` with weight+bias.
  Flow: `h = h + attn(norm1(h)); h = h + ff(norm2(h))`.
- D_Cortex `FusionBlock` = a StandardTransformerBlock (`norm_self`+`self_attn`,
  `norm_ff`+`ff`) PLUS a memory path (`norm_h`,`norm_mem`,`cross_attn`,`mem_gate`).
  Flow: `h = h + self_attn(norm_self(h)); h = h + gate*cross_attn(...); h = h + ff(norm_ff(h))`.
  With `cross_attn.out` zero-initialized the memory term is exactly 0, so a fusion
  block reduces to a standard pre-norm transformer block at init.
- GPT-2 is also PRE-NORM with `ln_1`/`ln_2` and combined-QKV `c_attn`. The attention
  weights live in `Conv1D` modules whose `weight` is stored transposed relative to
  `nn.Linear` ([in, out] vs [out, in]), so every Conv1D weight is TRANSPOSED on load.
  Biases copy directly.

## Layer selection (16 of 24, even spacing)

`indices = [round(i*24/16) for i in range(16)] = [0,2,3,4,6,8,9,10,12,14,15,16,18,20,21,22]`

- decoder `dec_standard_blocks[0..11]` <- GPT-2 layers [0,2,3,4,6,8,9,10,12,14,15,16]
- decoder `dec_fusion_blocks[0..3]`   <- GPT-2 layers [18,20,21,22]

Depth order is preserved (earlier GPT-2 layers -> earlier decoder blocks).

## Per-block mapping (each selected GPT-2 layer g -> decoder block d)

Standard block (d in 0..11) and fusion self-attn/FFN (d in 12..15):

| D_Cortex target | GPT-2 source | op |
|---|---|---|
| norm1 / norm_self .weight,.bias | ln_1 .weight,.bias | copy |
| attn/self_attn .qkv.weight (3072,1024) | attn.c_attn.weight (1024,3072) | transpose |
| attn/self_attn .qkv.bias (3072) | attn.c_attn.bias (3072) | copy |
| attn/self_attn .out.weight (1024,1024) | attn.c_proj.weight (1024,1024) | transpose |
| attn/self_attn .out.bias (1024) | attn.c_proj.bias (1024) | copy |
| norm2 / norm_ff .weight,.bias | ln_2 .weight,.bias | copy |
| ff.fc1.weight (4096,1024) | mlp.c_fc.weight (1024,4096) | transpose |
| ff.fc1.bias (4096) | mlp.c_fc.bias (4096) | copy |
| ff.fc2.weight (1024,4096) | mlp.c_proj.weight (4096,1024) | transpose |
| ff.fc2.bias (1024) | mlp.c_proj.bias (1024) | copy |

QKV ordering matches: GPT-2 `c_attn` emits [q|k|v] in 1024-chunks, head layout
(16x64); D_Cortex reshapes `qkv` as (3, n_heads, head_dim) -> same q,k,v order.

## Shared / global mapping

| D_Cortex target | GPT-2 source | op |
|---|---|---|
| shared_token_emb.weight (50257,1024) | wte.weight | copy (shared by enc+dec) |
| shared_pos_emb.weight[:1024] | wpe.weight (1024,1024) | copy; rows 1024..2047 left at model init |
| dec_final_norm.weight,.bias | ln_f.weight,.bias | copy |
| dec_lm_head.weight | (tied to shared_token_emb) | tie preserved, no separate copy |

## Zero-initialized (fresh fusion memory path made inert)

For each fusion block: `cross_attn.out.weight = 0`, `cross_attn.out.bias = 0`.
This makes the cross-attention output exactly 0, so the gated memory term
`gate * cross_attn(...)` is 0 at init and the fusion block is a no-op on memory.

## Left FRESH (normal D_Cortex init, trained later in the memory campaign)

- Entire encoder (Agent A): `encoder.*`.
- Memory addressing/reading: `shared_address_encoder`, `shared_query_engine`,
  `dec_state_reader`, `dec_episode_reader`, `dec_conflict_reader`,
  `dec_archive_reader`, `dec_working_reader`, `dec_read_fusion`.
- Fusion memory params not zeroed: `norm_h`, `norm_mem`, `cross_attn.q`,
  `cross_attn.kv`, `mem_gate` (irrelevant at init because `cross_attn.out`=0).
- `aux_answer_head`, `consolidator`.

## Scale-matched (derived from gpt2, not left at default)

- `dec_emb_norm`: an extra LayerNorm on the embeddings that GPT-2 has NO
  equivalent for (GPT-2 feeds raw wte+wpe, per-token std ~0.12, into block 0).
  At the default weight=1 it normalizes each token vector to unit std, inflating
  the residual-stream scale ~8.3x and wrecking the warm-start (measured: ppl
  133.65). FIX: `dec_emb_norm.weight` is set to the measured gpt2 embedding
  per-token std (~0.121, scalar) and `dec_emb_norm.bias` to 0, so
  `dec_emb_norm(emb) ~= emb` (scale-preserving). This drops warm-start ppl to
  46.95 (vs the 46.45 pure-pruning floor), i.e. it removes essentially all of
  the dec_emb_norm perturbation.

## Known drifts (reported, not hidden)

1. Activation: GPT-2 uses `gelu_new` (tanh approximation); D_Cortex `FeedForward`
   uses `nn.GELU()` (exact erf). Difference is < 0.1% per activation. Cannot be
   changed without modifying `dcortex/` (forbidden) and would not survive a
   state_dict round-trip anyway, so it is reported and kept exact.
2. Positions 1024..2047 of `shared_pos_emb` are fresh (gpt2-medium has only 1024).
   The campaign runs at context 1024, so they are unused for now.
3. `dec_emb_norm` has no GPT-2 source; it is scale-matched to the gpt2 embedding
   std (see "Scale-matched" above) rather than left at the inflating default.
4. Only 16 of 24 GPT-2 layers are used; an evenly-spaced layer subset is a pruned
   model whose adjacent-layer I/O does not compose perfectly, so warm-start ppl is
   expected ABOVE gpt2-medium's native ppl. Measured: pure 16-layer pruning floor
   = 46.45 ppl; warm-start = 46.95 ppl; gpt2-medium native (24 layers) = 27.99.
   The pruning is the entire residual gap; the campaign fine-tunes it away.
