# Role-binding evolution: content-addressed binding, held-out-construction benchmark

Engineering register. Claim status governs every line. Single environment, 10 seeds = MEASURED, not PROVEN.

## Diagnosis addressed
The positional binding head (BiGRU over position-marked role masks) passed seen syntax
(RB2 100%) but failed unseen syntax (RB3 56.9% exact, 31.4% confident wrong-mapping): it
learned a positional/lexical shortcut. This evolution refactors the binding layer above the
FROZEN substrate into a content-addressed, relation-typed, position-blind matcher and retests
on held-out syntactic constructions.

## Architecture (refactor, substrate untouched)
ContentAddressedRoleBinder (scripts/train_role_evolution.py): pools each entity/value token
span from the frozen decoder-standard states, builds a relation-keyed query per entity and a
relation-keyed key per value, binds by scaled dot-product matching (content-addressed, no
position features, one uniform bind for all families), and emits a learned abstain logit. Loss
adds a confident-wrong penalty so a wrong bind costs more than abstaining. The substrate
(warmstarted_init.pt) is read-only and never trained (G_SEALS).

## Data (family + entity holdout, non-trivial)
Corpus data/role_evolution/role_evolution_corpus.jsonl, SHA 0d58ed20b5fb57ad8df5c04d0e7365c058287c46e0fd3034eedb24d1282b2390, 422 records.
Entire entity pools AND entire cue constructions are held out: train uses copular cues
("X is the capital of Y"), validation possessive ("X's capital"), evaluation relative/passive/
concessive ("the city that B holds", "used by B", "answers to B"). RB4 audit: 9/9 PASS;
lexical/position baseline exact = 0.0% (non-trivial). Identity/
swapped labels ~50/50.

## Result (held-out constructions, 10 seeds)
| metric | evolved (median [min/max]) | RB3 prior |
|---|---|---|
| exact-match | 74.5% [56.9%/87.2%] | 56.9% |
| confident wrong-mapping | 13.7% [1.0%/28.4%] | 31.4% |
| abstain (known) | 8.8% | n/a |
| ambiguous abstain | 98.1% [26.9%/100.0%] | n/a |

Per held-out construction (median exact / wrong):
- eval_concessive: exact 80.9%, wrong 8.8%
- eval_passive: exact 77.9%, wrong 4.4%
- eval_relative: exact 63.2%, wrong 30.9%

## Gates (pre-declared, frozen)
- [PASS] G_GEN: held-out-family median exact-match 74.5% [56.9%/87.2%/std 0.080] over 10 seeds; floor 70%; lexical baseline 0.0%; uplift +74.5% (need >= 30%).
- [PASS] G_SAFE: held-out-family median wrong-mapping 13.7% [1.0%/28.4%]; ceiling 15% (RB3 was 31.4%); abstain median 8.8%.
- [PASS] G_CALIB: ambiguous-item abstain rate median 98.1% [26.9%/100.0%]; floor 70% (prefers abstain over confident wrong-bind on uncertain items).
- [PASS] G_SEALS: sealed substrate/semantic sources byte-identical=True; warmstart SHA 8eb5362ed39fd6b9... (read-only, never trained).

## Honest claim status
- MEASURED improvement: content-addressed binding generalizes to unseen cue constructions
  better than the positional head (74.5% vs 56.9% exact; wrong-mapping
  median 13.7% vs 31.4%, roughly halved), and the model abstains on
  uncertain items instead of confidently wrong-binding.
- PARTIAL and borderline: the relative-clause construction stays near RB3 level (high wrong-
  mapping), per-seed variance is wide (56.9%-87.2%), and
  the G_GEN median clears the 70% floor only narrowly. This is not a solved problem.
- The non-triviality constraint forces a shared crossing word order across families, so held-out
  diversity is in the cue PREDICATE, not word order; the head is position-blind, so this is the
  fair axis for it.
- Single environment, 10 seeds = MEASURED. >= 20 seeds + multi-environment required
  for legal-grade / patent use. Not PROVEN.
