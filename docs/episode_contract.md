# v11 DUAL_AGENT episode contract (PHASE A)

The exact schema the v11 structural-episode path expects, extracted from
`colab/step2_training_v11.py` (`run_structural_episode`, the episode generators,
`TrainConfig`) and `dcortex/model.py` (`encode`/`decode`). The adapter
(`scripts/episode_adapter.py`) emits episodes in THIS format. The `dcortex/`
architecture and the loss composition are NOT changed.

## Episode object (what the path consumes)

An episode is the v11 `EpisodeGT`:

| field | type | meaning |
|---|---|---|
| facts | list[FactInfo] | N facts written to memory this episode |
| prompt | str | the query text decoded against memory |
| target_fact_idx | int | which fact the query is about (0..N-1) |
| answer_token_id | int | single gpt2 token id the model must emit |
| ep_type | str | label ("cloze" for the adapter) |

`FactInfo`: `text` (str, the fact statement), `answer_token_id` (int, single
token used for lexical value binding), plus `entity`/`value` for provenance.

## Tokenization and shapes

- Tokenizer: tiktoken gpt2, vocab 50257, `EOT = enc.eot_token`.
- Fact write: `f_ids = _pad(encode_ordinary(fact.text) + [EOT], seq_len)`, with
  `seq_len = 64`; `_pad` truncates to seq_len or right-pads with EOT.
  Tensor `xf = [1, seq_len] long`. Call:
  `model.encode(xf, answer_token_id=[fact.answer_token_id], lexical_alpha=0.9,
  force_bank="working")` -> returns aux dict; `aux["w_k_ent"][0]` is the fact key.
- Query: `p_ids = encode_ordinary(prompt)`, `xp = [1, seq_len] long` (padded).
  `logits, retrieved = model.decode(xp, return_retrieved=True)`;
  `aux_logits = model.aux_answer_head(retrieved)` -> `[1, vocab]`.

## Loss (preserved exactly, not reinvented)

- `L_emit = cross_entropy(aux_logits, [answer_token_id])` (the masked answer
  loss: only the single answer token, via the aux head).
- `L_sel`: query key vs the N fact keys. `q_k_ent` from the query's address
  code, `K = normalize(stack(fact_keys))`, `sim = normalize(q_k_ent) @ K.t()`,
  `L_sel = -log_softmax(sim * 5.0)[target_fact_idx]` (retrieval supervision).
- `L_sep_neg`: off-diagonal fact-key similarity penalty
  `relu(sim_offdiag - 0.5)^2.mean()` (keys must be separable).
- `total = w_emit*L_emit + w_sel*L_sel + w_sep_neg*L_sep_neg`
  with `w_emit=1.0, w_sel=1.0, w_sep_neg=0.5`.

## Facts per episode

v11 synthetic episodes use 3..5 facts (`min_facts=3, max_facts=5`). The adapter
bundles K facts (config `EPISODE_FACTS`, default 6) with DISTINCT subjects, so
the query identifies which fact to retrieve. Bank capacity for the working bank
is `n_work_slots` (16 in BIG config); K must stay well under it.

## What "memory is required" means here (the cloze invariant)

The answer token must exist ONLY in the fact written to memory, NEVER in the
query. The adapter enforces this: the query is the fact clause truncated exactly
before the value token, the answer is the first gpt2 token of the value, and the
value is verified absent from the query text. Because the bundle holds K facts
with different subjects, the decoder must retrieve the correct fact (by subject)
to emit its value. Memory-zeroed cannot; memory-enabled can. That gap is the
ablation metric.

## The flag the adapter feeds

The DUAL_AGENT path generates episodes via `generate_episode()`. The adapter
provides an alternative episode source (an iterator over deserialized adapter
episodes). The training driver selects it with a flag; no model code changes.
