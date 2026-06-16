# Role-binding SCALE & CERTIFY: held-out-predicate benchmark

Engineering register. Claim status governs every line. MEASURED, not PROVEN.

## What was tested
The UNCHANGED ContentAddressedRoleBinder (no architecture change) trained over the FROZEN
substrate on a scaled multi-relation corpus, certified on GENUINELY held-out predicate types
and a held-out relation across 20 seeds plus a CPU second-environment repro.

## Integrity correction (reported, not hidden)
A first scale corpus held out only near-identical one-word variants of each construction (e.g.
"near A" vs "by A"), so every predicate type was effectively seen in training and the benchmark
returned a SPURIOUS 100%. That was caught and discarded. The result below is on the CORRECTED
corpus where ENTIRE predicate types are held out: training sees copular + possessive only;
evaluation predicates (concessive, relative) and the currency-code relation are NEVER in training.

## Data
Corpus data/role_scale/role_scale_corpus.jsonl, SHA 408d64a9b87baabb82c2bd8c3cdc63804cba1be3816dae607d4a16b7927e86bb, 2070 records, 3
pinned relations (samayo/country-json @41d4084, MIT): capital, abbreviation, currency_code.
Holdout: train = copular/possessive on (capital, abbreviation); validation = passive (held-out
predicate, early-stop); evaluation = concessive/relative on (capital, abbreviation, currency_code).
RB4 audit 9/9; lexical/position baseline exact 0.0%; labels ~50/50.

## Result (held-out predicates, 20 seeds)
- aggregate exact-match: 97.1% [95.1%/98.8%], std 0.011
- aggregate wrong-mapping: 2.8%; ambiguous abstain 100.0%; known abstain 0.0%
- per held-out construction:
  - concessive: exact 93.1%, wrong 6.5%
  - relative: exact 99.6%, wrong 0.4%
- per relation (currency_code = held-out relation):
  - abbreviation: exact 97.5%, wrong 2.3%
  - capital: exact 97.7%, wrong 2.0%
  - currency code: exact 96.8%, wrong 3.2%
- second environment (CPU, 8 seeds): median exact 97.1%, delta vs CUDA 0.1%

## Gates (pre-declared, frozen) - all pass
- [PASS] G_GEN: held-out median exact 97.1% (floor 75%); lexical 0.0%; uplift +97.1%.
- [PASS] G_PER_CONSTRUCTION: every held-out construction exact>=65% & wrong<=15%; failing=none; concessive:93%/7%; relative:100%/0%
- [PASS] G_SAFE: aggregate median wrong-mapping 2.8% (ceiling 12%; RB3 31.4%).
- [PASS] G_CALIB: ambiguous abstain median 100.0% (floor 70%); known abstain median 0.0%.
- [PASS] G_STABILITY: min exact 95.1% (floor 65%; RB3 56.9%) & std 0.011 (ceiling 0.07) over 20 seeds.
- [PASS] G_REPRO: second env (CPU, 8 seeds) median exact 97.1% vs ENV1 97.1%, delta 0.0% (tol 5%); same machine, CPU vs CUDA backend (NOT distinct hardware).
- [PASS] G_SEALS: sealed sources byte-identical=True; substrate SHA 8eb5362ed39fd6b9 read-only.

## Honest claim status
- Data-starved hypothesis CONFIRMED: scaling data (3 relations, 2 seen predicates, ~2070 records)
  lifted held-out exact 74.5% -> 97.1%, fixed the relative weak spot
  (evolution 63.2%/30.9% wrong -> 99.6%/0.4%), and crushed seed variance
  (min 56.9% -> 95.1%, std 0.011).
- SCOPED: these are simple, EXPLICITLY-CUED 2-entity 2-value bindings; held-out diversity is in
  the cue PREDICATE and the relation, within a shared crossing word order (non-triviality forces
  shared order; the head is position-blind, so predicate is the fair axis). This is NOT arbitrary
  syntactic structure, implicit relations, or multi-hop binding.
- Second environment is CPU vs CUDA on the SAME machine (not distinct hardware). True multi-
  hardware reproduction is still required for full legal-grade.
- MEASURED on the scoped task, single machine, 20 seeds + CPU/CUDA repro. NOT PROVEN
  for general systematic role-binding. Do not promote to PROVEN.
