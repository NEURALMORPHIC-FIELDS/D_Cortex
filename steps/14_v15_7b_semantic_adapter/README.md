# Step 14 — v15.7b — Conservative Semantic Hypothesis Adapter

**Status**: QUERY-SIDE SUBMILESTONE SEALED. End-to-end semantic-memory
integration remains open.

## Problem

Pas 7a can metabolize longitudinal evidence, but no explicit interface exists
for a semantic abstraction layer to submit hypotheses. F1 novel paraphrase,
F3 novel lexical alias, and F5 novel query forms therefore remain unresolved.
Directly connecting an uncertain semantic producer to committed memory would
violate the project's honesty and causality requirements.

## Architectural hypothesis

A strict adapter that accepts semantic hypotheses only as provisional evidence
or query-only interpretations, requires complete provenance, and emits a
deterministic audit record can connect future semantic producers to the sealed
consolidator without allowing semantic uncertainty to contaminate committed
memory.

## Contract

### Input: `SemanticHypothesis`

Every hypothesis must contain:

- stable `hypothesis_id`
- `episode_id`
- `mode`: `FACT` or `QUERY`
- original `source_text`
- producer identity and version
- provenance evidence identifiers
- confidence and uncertainty in `[0, 1]`
- optional entity, attribute, and value interpretation
- requested destination: `PROVISIONAL_ONLY`, `QUERY_ONLY`, or
  `COMMITTED_DIRECT`

### Output: `AdapterDecision`

The adapter returns exactly one decision:

- `ACCEPT_PROVISIONAL`: fact hypothesis is safe to expose to provisional memory
- `ACCEPT_QUERY`: query interpretation is safe for read-only use
- `REJECT`: hypothesis violates the contract

`COMMITTED_DIRECT` is always rejected. The adapter has no API capable of
writing committed memory.

### Consolidator-facing evidence

Accepted fact hypotheses become immutable `ProvisionalCandidate` records.
Distinct confirmation count is derived only from distinct `episode_id` values.
Repeated submissions inside one episode cannot inflate longitudinal evidence.
Conflicting values remain separate candidates for the same entity/attribute
slot.

## Frozen acceptance gates

| Gate | Requirement |
|---|---|
| G0_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |
| G1_NO_DIRECT_COMMIT | 100% direct-commit requests rejected |
| G2_PROVENANCE_REQUIRED | 100% missing-provenance hypotheses rejected |
| G3_QUERY_READ_ONLY | 100% query-write attempts rejected |
| G4_ANTI_INFLATION | repeated same-episode evidence counts as one confirmation |
| G5_LONGITUDINAL_CONFIRMATION | same hypothesis across two episodes counts as two confirmations |
| G6_CONFLICT_PRESERVATION | conflicting values remain separate |
| G7_DETERMINISTIC_AUDIT | repeated identical run produces byte-identical audit JSON |
| G8_ROUNDTRIP | serialized contract objects reconstruct exactly |
| G9_INVALID_RANGE_REJECTED | 100% invalid confidence/uncertainty values rejected |

All ten gates must pass. Thresholds are not relaxed after execution.

## Maintained invariants

- `steps/13_v15_7a_consolidation/code.py`: untouched
- Pas 6 critical path: untouched
- Pas 7a D6/D7/D8/D9 and Gates 0–9: untouched
- query path: untouched
- no hardcoded synonym map
- no automatic committed-memory write
- every accepted or rejected hypothesis is auditable

## Scope guard

This step defines and validates the adapter. It does not yet:

- generate semantic hypotheses from language
- improve F1/F3/F5
- modify parser behavior
- modify consolidator thresholds
- modify the neural memory architecture

## Successor

After all adapter gates pass, a separate semantic producer may be implemented.
Its output must pass through this adapter. The producer is evaluated on F1/F3/F5
while Pas 6/Pas 7a honesty and regression gates remain frozen.

## Successor opened: conservative prototype producer

The adapter gates passed in the local implementation session. The first
permitted producer is therefore a conservative latent-prototype matcher:

- embeddings are supplied by an explicit backend
- canonical prototypes are supplied by the caller
- no synonym map exists inside the producer
- each semantic axis requires both an absolute similarity threshold and a
  top-1 margin
- uncertain inputs produce abstention, not a guessed hypothesis
- every emitted hypothesis is submitted through
  `ConservativeSemanticAdapter`

Frozen producer gates:

| Gate | Requirement |
|---|---|
| P0_ADAPTER_REQUIRED | producer output is accepted only through the adapter |
| P1_NO_DIRECT_COMMIT | producer always requests provisional-only or query-only |
| P2_THRESHOLD_ABSTENTION | below-threshold axis causes abstention |
| P3_MARGIN_ABSTENTION | ambiguous top-1 margin causes abstention |
| P4_DETERMINISTIC | identical input/backend produces identical result |
| P5_NO_SYNONYM_MAP | producer implementation contains no lexical alias table |
| P6_QUERY_NOVEL_FORM | real warm-start probe reaches ≥60% attribute accuracy on frozen novel-query set |
| P7_AMBIGUOUS_HONESTY | real warm-start probe abstains on ≥80% frozen ambiguous queries |

P6/P7 are measurements of the producer probe only. They do not establish
F1/F3/F5 improvement until the official families are run.

## Producer iteration 1 verdict

The frozen warm-start token-mean probe failed `P6_QUERY_NOVEL_FORM`:

- total entity+attribute accuracy: `0/20`
- emitted hypotheses: `0/20`
- attribute top-1 identity before abstention: `20/20`
- entity top-1 identity before abstention: `15/20`
- ambiguous abstention: `10/10`

The producer remained honest, but its single global margin could not separate
novel semantic intent from ambiguity. Thresholds and gates remain unchanged;
this failed iteration is retained as the baseline.

## Successor opened: decoder-native causal-likelihood producer

The next permitted experiment uses the warm-start D_Cortex decoder as a
read-only semantic scorer. It ranks caller-supplied canonical candidates by
their conditional continuation likelihood. It does not write model memory,
does not contain synonym tables, and submits every emitted interpretation
through `ConservativeSemanticAdapter`.

The observed iteration-1 query set is now calibration/development data only.
A separate, newly frozen evaluation set governs the successor verdict.
Attribute abstention threshold selection is deterministic: choose the smallest
margin threshold that abstains on at least 80% of calibration ambiguous
queries. The evaluation threshold is never changed after the first evaluation.

Frozen causal-likelihood gates:

| Gate | Requirement |
|---|---|
| Q0_ADAPTER_REQUIRED | every emitted interpretation is accepted through the adapter |
| Q1_NO_DIRECT_COMMIT | every emitted interpretation is query-only |
| Q2_CALIBRATION_SEPARATION | calibration and evaluation query texts have zero exact overlap |
| Q3_DETERMINISTIC | repeated input/model produces an identical result |
| Q4_NO_SYNONYM_MAP | producer implementation contains no lexical alias table |
| Q5_NOVEL_QUERY_ACCURACY | held-out total entity+attribute accuracy is at least 75% |
| Q6_AMBIGUOUS_HONESTY | held-out ambiguous-query abstention is at least 80% |
| Q7_NON_TRIVIAL_LABEL_OVERLAP | exact canonical attribute-label overlap baseline is at most 10%, while Q5 passes |
| Q8_BASELINE_UPLIFT | held-out accuracy exceeds the frozen token-mean producer by at least 30 percentage points |
| Q9_READ_ONLY_MODEL_STATE | decoder scoring leaves all model buffers byte-identical |
| Q10_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |

Passing these gates would validate a conservative query-intent producer in one
local environment. It would still not establish official F1/F3/F5 improvement,
fact internalization, or semantic-memory advantage.

## Producer iteration 2 verdict

The first frozen causal-likelihood evaluation failed its coverage target while
preserving precision and honesty:

- held-out total accuracy: `19/32 = 59.4%` (`Q5` FAIL)
- emitted precision: `19/19 = 100%`
- held-out ambiguous abstention: `15/16 = 93.8%`
- exact-label baseline: `0/32`
- frozen token-mean baseline: `0/32`
- all model buffers remained byte-identical

The decoder-native scorer therefore contains useful non-trivial semantic
signal, but one prompt view is not sufficient to meet the frozen 75% coverage
gate. The gate and calibrated threshold remain unchanged.

## Successor opened: multi-view likelihood evidence fusion

The next experiment fuses multiple read-only likelihood views instead of
lowering the failed threshold. Each view independently ranks the same
caller-supplied canonical candidates. The aggregate choice is the mean
candidate probability and must receive explicit prompt-view consensus.

Frozen views:

- entity: `Answer entity:` and `This question is about`
- attribute: `Answer type:`, `This question asks about the object's`,
  `The requested attribute is`, and `Requested property:`
- entity consensus: `2/2`
- attribute consensus: at least `2/4`
- attribute margin calibration: smallest aggregate margin that abstains on at
  least 80% of all previously observed ambiguous development queries

The first verdict uses `n=500`, seed `20261215`, generated directly from the
sealed Pas 7a `F5_QUERY_FORMS`, `HOLDOUT_ENTITIES_SINGLE`, attribute types, and
attribute values extracted from its AST. It is a query-interpretation probe,
not an end-to-end F5 commit result.

The frozen ambiguous evaluation uses `n=100`, seed `20261216`, official
single-token holdout entities, and these attribute-unspecified forms:

1. `Give me a general overview of the {entity}.`
2. `What should be remembered about the {entity}?`
3. `Describe the {entity} without focusing on one property.`
4. `Tell me any relevant fact about the {entity}.`
5. `Summarize available information concerning the {entity}.`
6. `What is noteworthy about the {entity}?`
7. `Discuss the {entity} in broad terms.`
8. `Provide an unrestricted description of the {entity}.`
9. `What can be recalled regarding the {entity}?`
10. `Share something about the {entity}.`

Frozen multi-view gates:

| Gate | Requirement |
|---|---|
| R0_ADAPTER_REQUIRED | every emitted interpretation is accepted through the adapter |
| R1_NO_DIRECT_COMMIT | every emitted interpretation is query-only |
| R2_SEALED_F5_SOURCE | 500 queries come from AST-extracted sealed F5 definitions at the frozen seed |
| R3_CALIBRATION_SEPARATION | development/calibration and verdict query texts have zero exact overlap |
| R4_DETERMINISTIC | repeated input/model produces an identical result |
| R5_NO_SYNONYM_MAP | producer implementation contains no lexical alias table |
| R6_F5_QUERY_INTERPRETATION | total entity+attribute accuracy is at least 85% |
| R7_WRONG_INTERPRETATION | wrong emitted interpretations are at most 2% of all F5 trials |
| R8_AMBIGUOUS_HONESTY | ambiguous-query abstention is at least 80% |
| R9_TOKEN_MEAN_UPLIFT | accuracy exceeds the frozen token-mean producer by at least 30 percentage points |
| R10_MULTIVIEW_CONSENSUS | every emitted interpretation satisfies the frozen view-consensus counts |
| R11_READ_ONLY_MODEL_STATE | decoder scoring leaves all model buffers byte-identical |
| R12_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |

No threshold, prompt view, consensus count, seed, sample count, or gate is
changed after the first verdict run.

## Producer iteration 3 verdict

The frozen multi-view F5 query-intent verdict failed:

- total sealed-F5 interpretation accuracy: `317/500 = 63.4%`
- emitted interpretations: `335/500`
- wrong emitted interpretations: `18/500 = 3.6%`
- attribute-unspecified abstention: `85/100 = 85.0%`
- token-mean baseline: `0/500`

The failure is concentrated in the `state` attribute. Prompt-view fusion is a
useful read-only baseline, but prompt engineering is no longer an acceptable
next step.

## Successor opened: trained pooled semantic internalizer

The next experiment trains a small, separate semantic internalization head on
frozen D_Cortex token embeddings. The D_Cortex model and all memory buffers
remain frozen. The head receives five pooled views of the token embeddings:
mean, max, min, first token, and last token. Separate attribute and entity MLPs
avoid cross-task interference.

Training data is generated only from sealed standard `V15_QUERY_TEMPLATES` and
`V15_FACT_TEMPLATES`, plus previously observed ambiguous development texts and
non-attribute distractor sentences. Training excludes all F1, F3, and F5 query
forms. Attribute classes, including `UNKNOWN`, are balanced deterministically.

Frozen policy:

- head hidden width: `256`
- training seed: `20261220`
- attribute emission margin: `0.40`
- entity emission margin: `0.00`
- F1 verdict: `n=500`, seed `20261221`
- F3 verdict: `n=500`, seed `20261223`
- F5 verdict: `n=500`, seed `20261225`
- ambiguity verdict: `n=200`, seed `20261226`
- no threshold or training/evaluation definition changes after first verdict

Frozen trained-internalizer gates:

| Gate | Requirement |
|---|---|
| T0_TRAINING_SEPARATION | training contains no F1/F3/F5 query form and has zero exact overlap with verdict texts |
| T1_REAL_OPTIMIZATION | validation loss falls by at least 20% and best validation accuracy is at least 95% |
| T2_DETERMINISTIC_DATA | repeated dataset construction produces identical hashes and counts |
| T3_F1_QUERY_INTERPRETATION | F1 total entity+attribute accuracy is at least 85% |
| T4_F3_QUERY_INTERPRETATION | F3 total entity+attribute accuracy is at least 85% |
| T5_F5_QUERY_INTERPRETATION | F5 total entity+attribute accuracy is at least 85% |
| T6_WRONG_INTERPRETATION | wrong emitted interpretations are at most 2% in every family |
| T7_AMBIGUOUS_HONESTY | attribute-unspecified abstention is at least 80% |
| T8_ADAPTER_REQUIRED | every emitted interpretation is accepted through the adapter as query-only |
| T9_FROZEN_SUBSTRATE | D_Cortex parameters and buffers remain byte-identical; only the head is optimized |
| T10_BASELINE_UPLIFT | each family exceeds the frozen token-mean producer by at least 30 percentage points |
| T11_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |

Passing would establish a trained semantic query internalizer on three sealed
external language families in one local environment. It would still not
establish fact-side semantic internalization, end-to-end F1/F3/F5 commit
improvement, or semantic-memory advantage.

## Producer iteration 4 verdict

The standard-language-only trained internalizer produced a decisive partial
result:

- standard validation: `100%` joint accuracy, `99.4%` loss reduction
- F1 novel paraphrase: `236/500 = 47.2%`, wrong `1.6%`
- F3 novel lexical alias: `253/500 = 50.6%`, wrong `19.8%`
- F5 novel query form: `438/500 = 87.6%`, wrong `0.6%`
- attribute-unspecified abstention: `200/200`
- frozen substrate: byte-identical

This proves that standard facts and queries are sufficient for F5 form
generalization at the frozen margin, but not for F1/F3 semantic abstraction.
The next step must add semantic supervision without evaluating on forms seen in
training.

## Successor opened: leave-one-form-out semantic curriculum

F1 and F3 each contain four sealed query forms per attribute. Four folds are
frozen. In fold `i`, the internalizer trains on standard V15 data plus F1/F3
forms with indices other than `i`, then evaluates only form index `i` for every
attribute and every official single-token entity. Every form is therefore
evaluated exactly once while absent from its fold's training data.

A fifth final head trains on all F1/F3 curriculum forms and is evaluated on F5,
which remains absent from training. All folds use the existing frozen pooled
feature architecture, separate heads, training seed family, and `0.40`
attribute emission margin.

The final F5 set reuses the already frozen `n=500`, seed `20261225` definition.
The new ambiguity set uses `n=200`, seed `20261236`, and these forms:

1. `Give a broad account of the {entity} without choosing an aspect.`
2. `What general knowledge concerns the {entity}?`
3. `Explain the {entity} in an unrestricted manner.`
4. `Provide any overview of the {entity}.`
5. `What is generally known regarding the {entity}?`
6. `Speak about the {entity} without a specific question.`
7. `Offer background on the {entity}.`
8. `Describe whatever matters about the {entity}.`
9. `Recall general information about the {entity}.`
10. `Summarize the {entity} broadly.`

Frozen curriculum gates:

| Gate | Requirement |
|---|---|
| C0_FORM_HOLDOUT | every evaluated F1/F3 form is absent from its fold training data; exact overlap is zero |
| C1_REAL_OPTIMIZATION | every fold reduces validation loss by at least 20% and reaches at least 95% joint validation accuracy |
| C2_F1_OUT_OF_FOLD | aggregate F1 out-of-fold total accuracy is at least 85% |
| C3_F3_OUT_OF_FOLD | aggregate F3 out-of-fold total accuracy is at least 85% |
| C4_OUT_OF_FOLD_WRONG | wrong emitted interpretations are at most 2% for both F1 and F3 |
| C5_F5_FINAL | final-head F5 total accuracy is at least 85% and wrong emitted interpretations are at most 2% |
| C6_AMBIGUOUS_HONESTY | new attribute-unspecified abstention is at least 80% |
| C7_ADAPTER_REQUIRED | every emitted interpretation is accepted through the adapter as query-only |
| C8_FROZEN_SUBSTRATE | substrate state remains byte-identical and only heads are optimized |
| C9_DETERMINISTIC | repeated curriculum construction yields identical hashes and counts |
| C10_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |

Passing would show that learned semantic supervision generalizes to held-out
forms inside F1/F3 while preserving F5 and honesty. It would not prove
open-domain semantics or end-to-end memory improvement.

## Producer iteration 5 verdict

The pooled leave-one-form-out curriculum produced a strong but incomplete
result:

- F1 out-of-fold: `369/528 = 69.9%`, wrong `0.0%` — FAIL
- F3 out-of-fold: `463/528 = 87.7%`, wrong `0.4%` — PASS
- final-head F5: `94.2%`, wrong `0.0%` — PASS
- new ambiguity abstention: `200/200`
- all holdout, optimization, adapter, determinism, substrate, and seal gates:
  PASS

The remaining failure is specifically syntactic. Five-view token pooling is
order-insensitive, so it cannot reliably represent unseen F1 constructions.

## Successor opened: frozen contextual syntax internalizer

The next experiment replaces only the feature backend. It runs the frozen
D_Cortex decoder standard blocks without memory reads or writes, then applies
the same five pooling views to contextual hidden states. The semantic head,
curriculum folds, margin `0.40`, gates, adapter policy, and sealed sources remain
unchanged.

Frozen contextual gates:

| Gate | Requirement |
|---|---|
| D0_FORM_HOLDOUT | every evaluated F1/F3 form remains absent from its fold training data |
| D1_F1_CONTEXTUAL_OUT_OF_FOLD | aggregate contextual F1 out-of-fold accuracy is at least 85% |
| D2_F3_CONTEXTUAL_OUT_OF_FOLD | aggregate contextual F3 out-of-fold accuracy is at least 85% |
| D3_F5_CONTEXTUAL_FINAL | final contextual F5 accuracy is at least 85% |
| D4_WRONG_INTERPRETATION | wrong emitted interpretations are at most 2% for F1, F3, and F5 |
| D5_AMBIGUOUS_HONESTY | ambiguity abstention remains at least 80% |
| D6_ADAPTER_REQUIRED | every emitted interpretation is accepted through the adapter as query-only |
| D7_FROZEN_SUBSTRATE | contextual decoder state remains byte-identical and only heads are optimized |
| D8_DETERMINISTIC | repeated curriculum construction yields identical hashes and counts |
| D9_MEMORY_BYPASS | contextual feature extraction performs no memory read or write |
| D10_SEALED_UNTOUCHED | Pas 7a sealed source SHA-256 remains unchanged |

This successor is justified by the pooled F1 failure and a development-only
contextual feasibility diagnostic. Its F1 forms are therefore not an untouched
independent holdout; any pass is a measured architecture regression result, not
proof of open-domain syntactic abstraction.

## Query-side contextual seal outcome

The first frozen contextual run passed all 11 D-gates:

- F1 contextual out-of-fold: `451/528 = 85.4%`
- F3 contextual out-of-fold: `474/528 = 89.8%`
- F5 contextual final: `463/500 = 92.6%`
- wrong emitted: F1 `0.0%`, F3 `0.0%`, F5 `0.2%`
- ambiguity abstention: `200/200`
- substrate and Pas 7a: byte-identical / untouched

This seals the query-side adapter plus contextual internalizer submilestone.
See `SEAL.md`. Fact-side internalization and end-to-end memory integration are
explicitly not sealed.
