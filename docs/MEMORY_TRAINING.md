# Memory campaign: warm-starting from gpt2-medium

This documents how the D_Cortex memory campaign uses the gpt2-medium warm-start
produced by `scripts/warmstart_gpt2.py`. It does not run the campaign.

## What the warm-start gives you

`runs/warmstart/warmstarted_init.pt` is a `DCortexV2Model` at the campaign BIG
config (hidden 1024, 16 heads, ff 4096, dec 16L, enc 4L, fusion 4, seq 2048)
whose decoder is initialized from pretrained gpt2-medium:

- 12 standard decoder blocks + the self-attn/FFN of the 4 fusion blocks are
  mapped from 16 evenly-spaced gpt2-medium layers (Conv1D weights transposed).
- token embedding, positional embedding (first 1024), and the final norm mapped.
- `dec_emb_norm` scale-matched to the gpt2 embedding scale.
- fusion memory cross-attention output zero-initialized (memory is inert at init).
- the encoder (Agent A), memory banks, addressing, readers, and the cross-attn
  query/key/value stay at fresh init.

Verified (PHASE C, `runs/warmstart/warmstart_verdict.json`): immediate held-out
perplexity 46.95 with NO training (random init was 59193.72; gpt2-medium native
is 27.99), and the zero-init fusion is provably inert (logits change by 0.0 when
memory is populated).

## How the memory campaign consumes it

The memory campaign loads `warmstarted_init.pt` as its INITIAL model state, then
trains the FULL graph. Unlike the LM_DECODER backbone run (`colab/train_campaign.py`,
which trained only the decoder from scratch and left Agent A and the memory banks
untrained, producing a word-salad backbone), this campaign trains Agent A + the
memory banks + the decoder jointly. The gpt2-derived decoder blocks fine-tune; the
encoder, memory banks, fusion gate, and memory cross-attention train from their
fresh init, so the model learns to actually read and write memory on top of a
competent language decoder instead of a from-scratch one.

Concretely, a memory-training script (the DUAL_AGENT structural path) should:

1. Build `DCortexV2Model(big_config())` and load the warm-start before any step:

   ```python
   ckpt = torch.load("runs/warmstart/warmstarted_init.pt", weights_only=False)
   model = DCortexV2Model(big_config())
   model.load_state_dict(ckpt["model"])
   model = model.to("cuda")
   ```

2. Train every parameter (do NOT freeze the encoder; this is the joint memory
   campaign, not the decoder-only backbone). Keep bf16 autocast, NO GradScaler,
   gradient checkpointing on, and the < 14.0 GB VRAM ceiling (peak grows once the
   AdamW states allocate, so size batch/context with ~2 GB of headroom).

3. Feed DUAL_AGENT episodes (facts written to memory via `encode(...)`, a query
   answered via `decode(...)`, answer-token loss + selection + separation losses,
   as in the v11 structural path), so gradients flow into Agent A and the memory
   banks. The held-out answer-token cross-entropy with memory reads ENABLED vs
   ZEROED is the decisive check that the memory path is trained and used.

Recommended source for the episode supply: an instructional file under
`E:\DATA\training_data_clean` (e.g. `06_FINAL_TRAINING_READY-003_BYON.jsonl`),
adapted into episodes. Budget: the warm-start removes the cold-start phase, so the
joint campaign can spend its step budget on learning memory behavior rather than
basic language modeling.

## Data safety

`warmstarted_init.pt` and the gpt2-medium HuggingFace cache are local and
gitignored (`*.pt`, `runs/`). Neither the corpus nor any derived weights are
committed.
