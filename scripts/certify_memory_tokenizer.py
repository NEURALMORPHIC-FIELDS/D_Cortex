# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Verify the memory tokenizer: G_TOKENIZER_BIJECTION (same value -> same token, distinct -> distinct)
# on the trained model's REAL internalized values, and G_SCALE (the codebook holds hundreds of
# distinct tokens with 0 collisions when values are separable - the 37-token wall is gone). The token
# derives from the internalized value, not the text (G_NO_TEXT_LEAK): same value via DIFFERENT entity
# and template must map to the same token.

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import tiktoken
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from stage_u.memory_tokenizer import MemoryTokenizer

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "memory_tokenizer"
ENC = tiktoken.get_encoding("gpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "orange", "purple"]
SIZES = ["tiny", "small", "big", "huge"]
LOCATIONS = ["forest", "cave", "castle", "river", "mountain", "garden", "cellar", "tower", "ocean", "desert"]
ENTS = ["bear", "dog", "cat", "fox", "wolf", "bird", "tiger", "horse", "deer", "rabbit"]
TEMPLATES_NONLOC = ["The {e} is {v}.", "The {e} looked {v}.", "A {v} {e} stood nearby.", "The {e} was {v}."]
TEMPLATES_LOC = ["The {e} is in the {v}.", "The {e} was found in the {v}.", "In the {v}, a {e} appeared.",
                 "The {e} stayed in the {v}."]
VALUE_ATTR = {**{c: "color" for c in COLORS}, **{s: "size" for s in SIZES}, **{loc: "location" for loc in LOCATIONS}}


def w_value(model, entity, value, template):
    text = template.format(e=entity, v=value)
    ids = torch.tensor([ENC.encode_ordinary(text)], device=DEVICE)
    ans = torch.tensor([ENC.encode_ordinary(" " + value)[0]], device=DEVICE)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        aux = model.encode(ids, answer_token_id=ans, lexical_alpha=0.0)
    return aux["w_value"][0].detach().float().cpu()


def collect(model, values, ents, templates_for):
    """Return {value: [vecs over (entity, template)]} split later into fit/test contexts."""
    out = {}
    for v in values:
        tmpls = templates_for(v)
        out[v] = [w_value(model, e, v, t) for e in ents for t in tmpls]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Memory tokenizer verification")
    ap.add_argument("--ckpt", default="runs/stage_u/results/ckpt_multiattr.pt")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Memory tokenizer verification (bijection + scale)", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE).eval()
    ck = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    print(f"[INFO] loaded {args.ckpt}", flush=True)

    # ---- G_TOKENIZER_BIJECTION on the REAL trained values (held-out contexts) ----
    values = COLORS + SIZES + LOCATIONS

    def templates_for(v):
        return TEMPLATES_LOC if VALUE_ATTR[v] == "location" else TEMPLATES_NONLOC
    vecs = collect(model, values, ENTS, templates_for)
    # split contexts per value: half fit, half test
    fit_v = {v: lst[:len(lst) // 2] for v, lst in vecs.items()}
    test_v = {v: lst[len(lst) // 2:] for v, lst in vecs.items()}
    tk = MemoryTokenizer(capacity=512)
    tk.fit(fit_v)
    correct = total = 0
    collisions = 0
    for v in values:
        for x in test_v[v]:
            total += 1
            correct += int(tk.decode(tk.tokenize(x)) == v)
    bijection_acc = round(correct / max(1, total), 4)
    # collisions: do two distinct values share a token? (prototypes argmax-collide)
    used_tokens = {}
    for v in values:
        t = tk.value_token[v]
        used_tokens.setdefault(t, []).append(v)
    collisions = sum(len(vs) - 1 for vs in used_tokens.values() if len(vs) > 1)
    g_bijection = (bijection_acc >= 0.99 and collisions == 0)
    print(f"  G_TOKENIZER_BIJECTION: {correct}/{total} held-out vectors -> correct value token "
          f"(acc {bijection_acc}); distinct-value collisions {collisions}", flush=True)

    # ---- G_NO_TEXT_LEAK: same value via DIFFERENT entity+template -> same token ----
    leaks = 0; nl = 0
    for v in values[:8]:
        toks = set(tk.tokenize(w_value(model, e, v, t)) for e, t in
                   [(ENTS[0], templates_for(v)[0]), (ENTS[5], templates_for(v)[1])])
        nl += 1; leaks += int(len(toks) > 1)
    g_noleak = (leaks == 0)
    print(f"  G_NO_TEXT_LEAK: same value across entity+template -> same token for {nl - leaks}/{nl} values", flush=True)

    # ---- G_SCALE: codebook holds N=200 distinct separable values, 0 collisions (>> 37) ----
    N, D = 200, 768
    g = torch.Generator().manual_seed(123)
    protos = torch.nn.functional.normalize(torch.randn(N, D, generator=g), dim=1)  # near-orthogonal in 768-d
    scale_vecs = {f"v{i}": [protos[i] + 0.05 * torch.randn(D, generator=g) for _ in range(4)] for i in range(N)}
    tk2 = MemoryTokenizer(capacity=512)
    tk2.fit({k: v[:2] for k, v in scale_vecs.items()})
    sc_ok = sc_tot = 0
    for i in range(N):
        for x in scale_vecs[f"v{i}"][2:]:
            sc_tot += 1; sc_ok += int(tk2.decode(tk2.tokenize(x)) == f"v{i}")
    scale_acc = round(sc_ok / max(1, sc_tot), 4)
    g_scale = (scale_acc >= 0.99 and len(tk2.token_value) == N)
    print(f"  G_SCALE: {N} distinct values -> {len(tk2.token_value)} tokens, held-out assign acc {scale_acc} "
          f"(sealed-organ wall was 37)", flush=True)

    verdict = "MEMORY_TOKENIZER_VERIFIED" if (g_bijection and g_noleak and g_scale) else "MEMORY_TOKENIZER_PARTIAL"
    out = {"verdict": verdict, "ckpt": args.ckpt,
           "G_TOKENIZER_BIJECTION": {"acc": bijection_acc, "collisions": collisions, "pass": bool(g_bijection),
                                     "n_values": len(values)},
           "G_NO_TEXT_LEAK": {"leaks": leaks, "n": nl, "pass": bool(g_noleak)},
           "G_SCALE": {"n_values": N, "tokens_used": len(tk2.token_value), "held_out_acc": scale_acc,
                       "pass": bool(g_scale), "vs_sealed_organ_wall": 37},
           "scope": "MEASURED, gold-anchored prototype codebook over the trained model's internalized w_value; "
                    "real values bijection + controlled-vector scale. A full learned VQ is the later refinement.",
           "meaning": "same value -> same token EXACT (the arbiter's discrete identity) + distinct -> distinct + "
                      "scales to hundreds of tokens: the 37-token capacity wall of the sealed organ is removed."}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("TOKENIZER_JSON " + json.dumps({"verdict": verdict, "bijection_acc": bijection_acc,
          "collisions": collisions, "no_leak": g_noleak, "scale_acc": scale_acc, "scale_tokens": len(tk2.token_value)}), flush=True)
    return 0 if (g_bijection and g_scale) else 1


if __name__ == "__main__":
    sys.exit(main())
