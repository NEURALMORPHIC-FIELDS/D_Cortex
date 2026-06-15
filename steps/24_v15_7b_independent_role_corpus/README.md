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
