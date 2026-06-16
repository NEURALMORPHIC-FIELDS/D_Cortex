# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Capable instruct base-model runtime for the professional control layer. Loads
# Qwen2.5-7B-Instruct in 4-bit NF4 (bitsandbytes, double quant, bf16 compute) with a
# named fallback order (Mistral-7B-Instruct-v0.3, then Llama-3.1-8B-Instruct). Exposes
# (a) unconstrained instruct generation (the RAW model, which produces PLAUSIBLE wrong
# answers on uncovered slots), (b) CONSTRAINED generation forcing the committed value
# at the factual slot via logit masking, and (c) layer -1 span-pooled hidden states
# (bf16) for a FRESH binder reinitialized on this model's geometry. Weights read-only.

import contextlib
import io
from dataclasses import dataclass
from typing import List, Optional

import torch

# fixed model order (validity-critical; do not substitute a sub-3B model)
PRIMARY = "Qwen/Qwen2.5-7B-Instruct"
FALLBACKS = ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B-Instruct"]


@dataclass
class ConstrainedResult:
    text: str
    forced_value: str
    unconstrained_slot_text: str
    overridden: bool


class QwenBaseModel:
    """Capable instruct LM in 4-bit NF4 with constrained decoding and hidden states."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.available = False
        self.reason = ""
        self.precision = "4bit-nf4"
        self.device = torch.device(device or "cuda")
        self.model = None
        self.tok = None
        self.model_name = ""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            self.reason = f"transformers/bitsandbytes unavailable: {exc}"
            return
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        order = [model_name] if model_name else [PRIMARY] + FALLBACKS
        for name in order:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tok = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForCausalLM.from_pretrained(
                        name, quantization_config=quant, device_map={"": 0},
                        torch_dtype=torch.bfloat16)
                    model.eval()
                self.model, self.tok, self.model_name = model, tok, name
                self.available = True
                break
            except Exception as exc:  # noqa: BLE001
                self.reason = f"load {name} failed: {type(exc).__name__}: {str(exc)[:160]}"

    def _chat_ids(self, user: str) -> List[int]:
        msgs = [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def generate_unconstrained(self, user_prompt: str, max_new_tokens: int = 40) -> str:
        ids = torch.tensor([self._chat_ids(user_prompt)], device=self.model.device)
        attn = torch.ones_like(ids)
        out = self.model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def _next_logits(self, ids: List[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=self.model.device)
        return self.model(x).logits[0, -1].float()

    @torch.no_grad()
    def generate_constrained(self, user_prompt: str, forced_value: str) -> ConstrainedResult:
        """Force the committed value at the answer slot by masking all other tokens."""
        base = self._chat_ids(user_prompt)
        value_ids = self.tok.encode(" " + forced_value, add_special_tokens=False)
        unconstrained: List[int] = []
        emitted: List[int] = []
        for tok_id in value_ids:
            logits = self._next_logits(base + emitted)
            unconstrained.append(int(logits.argmax().item()))
            emitted.append(tok_id)
        forced_text = self.tok.decode(emitted).strip()
        uncon_text = self.tok.decode(unconstrained).strip()
        return ConstrainedResult(text=forced_text, forced_value=forced_value,
                                 unconstrained_slot_text=uncon_text,
                                 overridden=uncon_text != forced_text)

    @torch.no_grad()
    def classify(self, user_prompt: str, options: List[str], answer_prefix: str = ""):
        """Constrained closed-set classification: score each option by its mean token
        log-probability as the continuation of the (chat) prompt and return the argmax.
        The model can only choose among `options` (logit-masking by construction)."""
        base = self._chat_ids(user_prompt)
        if answer_prefix:
            base = base + self.tok.encode(answer_prefix, add_special_tokens=False)
        sep = "" if (answer_prefix and answer_prefix.endswith(" ")) else " "
        best, best_lp, scores = None, None, {}
        for opt in options:
            opt_ids = self.tok.encode(sep + opt, add_special_tokens=False)
            ids = base + opt_ids
            logits = self.model(torch.tensor([ids], device=self.model.device)).logits[0].float()
            lp = 0.0
            for j, tid in enumerate(opt_ids):
                logp = torch.log_softmax(logits[len(base) + j - 1], dim=-1)
                lp += float(logp[tid])
            lp /= max(1, len(opt_ids))
            scores[opt] = round(lp, 4)
            if best_lp is None or lp > best_lp:
                best, best_lp = opt, lp
        return best, scores

    @torch.no_grad()
    def span_features(self, text: str, phrases: List[str]) -> Optional[torch.Tensor]:
        """Mean-pooled LAYER -1 hidden state per phrase span (bf16 -> float)."""
        try:
            enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
        except Exception:  # noqa: BLE001
            return None
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1][0].float()
        low = text.lower()
        pooled = []
        for phrase in phrases:
            start = low.find(phrase.lower())
            if start < 0:
                return None
            end = start + len(phrase)
            idx = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            if not idx:
                return None
            pooled.append(hidden[idx].mean(dim=0))
        return torch.stack(pooled, dim=0)

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size) if self.available else 0
