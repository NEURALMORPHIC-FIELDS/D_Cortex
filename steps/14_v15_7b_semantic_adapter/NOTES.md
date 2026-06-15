# NOTES — Step 14 v15.7b Conservative Semantic Hypothesis Adapter

## 2026-06-15 — Autonomous architectural decision

The canonical documentation was reviewed before implementation.

Decisive constraints:

- `MISIUNEA.txt`: semantic uncertainty must not contaminate stable memory.
- `docs/PROGRESS.md`: Pas 7b requires an explicit adapter and must not become a
  hardcoded synonym parser.
- `paper/D_CORTEX_PAS7A_SEAL.md`: Pas 7a, query path, and Gates 0–9 are sealed.
- The hardened neural-memory benchmark showed that prefix-cloze evaluation is
  lexically solvable and cannot establish semantic-memory advantage.

Decision:

Build the adapter contract first. Semantic producers remain untrusted. They may
submit hypotheses, but the adapter can expose fact hypotheses only as
provisional evidence and query hypotheses only as read-only interpretations.
There is deliberately no direct-commit API.

The adapter gate suite is frozen in `README.md` before code is written.

## 2026-06-15 — Adapter gates passed; producer scope opened

Current-session adapter verifier output: all ten gates PASS. Pas 7a SHA-256
matched the frozen value. Direct commit, missing provenance, and query writes
were rejected in every tested case.

The next component is a conservative latent-prototype producer. It is not a
parser replacement and contains no synonym map. It may abstain. Any emitted
hypothesis remains untrusted until the adapter accepts it.

## 2026-06-15 — Token-mean producer failed honestly

Current-session frozen probe result:

- `P0`–`P5`: PASS
- `P6_QUERY_NOVEL_FORM`: FAIL, `0/20`, zero emissions
- `P7_AMBIGUOUS_HONESTY`: PASS, `10/10` abstentions

The failure is representational. The warm-start token-mean backend selected the
correct attribute top-1 for `20/20` novel queries and the correct entity top-1
for `15/20`, but attribute margins were too close to ambiguous-query margins.
Relaxing the frozen threshold after observing this result is forbidden.

An exploratory, non-verdict diagnostic found that decoder-native conditional
likelihood contains a stronger query-intent signal than mean embeddings. The
next iteration is therefore a separate causal-likelihood producer with newly
frozen gates and a new held-out evaluation set. The old query set is explicitly
demoted to calibration/development data.

## 2026-06-15 — Single-view likelihood producer failed coverage

Current-session frozen verdict:

- `Q5_NOVEL_QUERY_ACCURACY`: FAIL, `19/32 = 59.4%`
- `Q7_NON_TRIVIAL_LABEL_OVERLAP`: FAIL because Q5 did not pass
- all other gates: PASS
- emitted interpretations: `19/19` correct
- ambiguous abstention: `15/16`
- token-mean baseline: `0/32`

The result supports a narrower claim: the D_Cortex decoder contains a useful
query-intent signal and can expose it conservatively, but a single prompt view
does not provide enough coverage. Threshold relaxation is not allowed.

Development-only analysis showed that averaging multiple independent
likelihood views can recover coverage while retaining ambiguity abstention.
This does not count as a verdict. A separate multi-view producer and a verdict
directly sourced from sealed F5 definitions are frozen before implementation.

## 2026-06-15 — Multi-view prompt fusion failed; training justified

Current-session frozen multi-view verdict:

- `R6_F5_QUERY_INTERPRETATION`: FAIL, `317/500 = 63.4%`
- `R7_WRONG_INTERPRETATION`: FAIL, `18/500 = 3.6%`
- all other R gates: PASS

The state family caused most wrong interpretations. Continuing to tune prompt
views would optimize the interface rather than build semantic internalization.

A development-only feasibility check trained a balanced pooled classifier using
only standard V15 facts and queries. It exceeded the 85% F5 target on the
already observed development F5 sample while abstaining on observed ambiguous
queries. This is not a verdict, but it is sufficient evidence to justify one
formal trained-internalizer cycle. F1 and F3 remain unmeasured holdout families.

## 2026-06-15 — Standard-only internalizer passed F5, failed F1/F3

Current-session frozen trained-internalizer verdict:

- F1: FAIL, `47.2%`
- F3: FAIL, `50.6%`, wrong emitted `19.8%`
- F5: PASS, `87.6%`, wrong emitted `0.6%`
- ambiguity honesty: PASS, `100%`
- substrate remained byte-identical

The result falsifies the hypothesis that standard V15 language alone is enough
for general semantic abstraction. It supports a narrower claim: the trained
head can generalize query form when the underlying concepts are represented in
standard facts, but lexical alias and paraphrase abstraction require semantic
supervision.

The successor is a strict leave-one-form-out curriculum. It is not a synonym
table: forms become supervised examples, and each reported form is evaluated
only in a fold where that exact form was absent from training.

## 2026-06-15 — Pooled curriculum passed F3/F5, isolated F1 syntax failure

Current-session curriculum verdict:

- F1 out-of-fold: FAIL, `69.9%`, wrong `0.0%`
- F3 out-of-fold: PASS, `87.7%`, wrong `0.4%`
- F5 final: PASS, `94.2%`, wrong `0.0%`
- ambiguity honesty: PASS, `100%`

The semantic curriculum solved lexical alias generalization without a synonym
table. The remaining F1 failure matches the architecture: mean/max/min/edge
pooling over raw token embeddings does not preserve token order.

A development-only diagnostic replaced raw pooling with frozen decoder
contextual states and reached `88.6%` F1 out-of-fold accuracy with zero wrong
emissions on the already observed folds. This justifies a formal contextual
backend implementation, but makes the next F1 result a regression measurement,
not an independent holdout proof.

## 2026-06-15 — Contextual query-side submilestone sealed

Current-session contextual verdict timestamp:
`2026-06-15T18:57:17+03:00`.

All 11 contextual gates passed. The contextual backend restored order-sensitive
F1 performance while preserving F3, F5, ambiguity honesty, adapter routing,
memory bypass, and substrate immutability.

The seal is intentionally narrow. The producer is not connected to Pas 7a
memory mutation, and no end-to-end semantic-memory claim is made.
