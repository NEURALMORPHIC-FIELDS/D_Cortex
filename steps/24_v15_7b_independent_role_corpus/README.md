# D_Cortex v15.7b-RB4 — Independent Role-Binding Corpus Gate

**Status: DATA PREREQUISITE FROZEN; NO MODEL CODE JUSTIFIED**
**Frozen at: 2026-06-15T20:23:00+03:00**

## Purpose

Prevent further development overfitting after RB3 showed unsafe
leave-one-syntax-family-out behavior. Before another role-binding model or any
memory ingestion, D_Cortex requires an independently sourced and auditable
role-binding corpus.

## Required corpus contract

- source constructions must not be derived from RB0/RB1/RB2/RB3 templates
- every record must expose:
  - source text
  - candidate entity inventory
  - candidate value inventory
  - attribute
  - exact one-to-one mapping or explicit ambiguity
  - provenance
- construction-family identity must be explicit and auditable
- train/validation/evaluation construction families must be disjoint
- exact text and normalized structural duplicates must be disjoint
- lexical and positional baselines must remain non-trivial
- ambiguity must have no committed truth mapping
- model training is forbidden until the corpus gate passes

## Frozen gates

| Gate | Requirement |
|---|---|
| N0_PREDECESSORS_PRESERVED | RB0–RB3 artifacts unchanged |
| N1_PROVENANCE | every record has a verifiable independent source |
| N2_CONSTRUCTION_SEPARATION | evaluation constructions absent from train and validation |
| N3_LABEL_STRUCTURE | known records have one exact one-to-one mapping; ambiguous records have none |
| N4_NO_DUPLICATE_LEAKAGE | no exact or normalized structural overlap across splits |
| N5_NON_TRIVIAL_BASELINES | no lexical/position baseline reaches 50% exact with safe ambiguity |
| N6_AMBIGUITY_AUDIT | ambiguity labels and inventories are internally consistent |
| N7_DATA_ONLY | no model training or threshold calibration performed |
| N8_SEALS_UNTOUCHED | Pas 7a and sealed semantic artifacts unchanged |

## Decision

Do not integrate RB2 into memory and do not tune RB3 on the same four syntax
families. The next executable work is corpus acquisition and audit only.

## Executable audit harness

Script:

```text
scripts/independent_role_corpus_audit.py
```

Default command:

```text
python scripts/independent_role_corpus_audit.py --corpus data/rb4/independent_role_corpus.jsonl
```

The default corpus path is intentionally not auto-created. A missing corpus is
a hard failure with instructions; it is not replaced with placeholder or
synthetic data.

## Frozen input schema

Each JSON/JSONL record must contain:

```json
{
  "record_id": "source-stable-id",
  "split": "train|validation|evaluation",
  "construction_family": "independent-family-name",
  "source_text": "source sentence or excerpt",
  "attribute": "attribute name",
  "entities": ["entity A", "entity B"],
  "values": ["value A", "value B"],
  "expected": [
    ["entity A", "attribute name", "value A"],
    ["entity B", "attribute name", "value B"]
  ],
  "ambiguous": false,
  "provenance": {
    "source_id": "document-id",
    "citation": "verifiable source reference"
  }
}
```

For ambiguous records, `expected` must be empty and `ambiguous` must be true.

## Frozen thresholds

- minimum total records: 120
- minimum records per split: 20
- minimum evaluation known records: 40
- minimum evaluation ambiguous records: 10
- maximum exact rate for lexical/position baselines: `< 50%`
- maximum overcommit rate for "safe ambiguity" baseline eligibility: `<= 2%`

## Claim guard

Passing this audit means only that the supplied corpus meets the RB4 data gate.
It does not mean RB2/RB3 generalizes, does not permit automatic memory
integration, and does not prove end-to-end D_Cortex improvement.
