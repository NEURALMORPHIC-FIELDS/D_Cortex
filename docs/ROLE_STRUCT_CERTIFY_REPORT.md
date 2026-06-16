# Role-binding STRUCTURAL certification (vnext3): cue-predicate vs genuine structure

Engineering register. Claim status governs every line. MEASURED at legal-grade rigor, NOT PROVEN.

Simple memory uplift was already surpassed and is not the current proof target. RB4 non-lexical
role binding is the current proof target. The claim is scoped to exactly the structural variation
that was verified, not beyond it.

## What this run adds over the prior scale certification
The prior scale run (97.1%) generalized to unseen cue PREDICATES, but the non-triviality crossing
order forced a shared surface structure, so it did not test STRUCTURAL generalization. This run
separates two axes and adds a genuine structural holdout, a no-memory causal control, a
compositional holdout, a pilot that calibrates the never-measured thresholds, a failure taxonomy,
and seed-variance localization. The UNCHANGED ContentAddressedRoleBinder is reused over the FROZEN
substrate.

## Corpus and validity (the make-or-break)
Main corpus data/role_struct/role_struct_corpus.jsonl SHA 53a032aedc9bdcf2ffd75c8c1a88cd3d74b95b416041042819c263c4362d48c6, 2685 records;
entity-disjoint calibration role_struct_calibration.jsonl SHA d1722a55c24c59ac6072a7ab9e3ee7ce9e4c6d44fd7a2c714590531e0b28f319.
Three pinned relations (samayo/country-json @41d4084, MIT): capital, abbreviation, currency_code.
Validator (data/role_struct/validation.json) all checks PASS:
- per-cell lexical/position baseline = 0.0% on every one of 13 relation x construction cells.
- entity pools GLOBALLY disjoint across train/validation/calibration/evaluation (no entity leakage).
- zero train<->eval shared (entity,value) facts; zero exact source-text duplicates across splits.
- structural distinctness verified: local gap ~10 tokens / depth 0; long_gap gap ~47 tokens;
  embedded depth ~8. long_gap min gap (40) > local max (17); embedded min depth (7) > local max (1).
RB4 audit on the main corpus: N5 non-triviality PASS (0.0%), N1/N3/N6/N7/N8(seals) PASS. N0 (RB0-RB3
lineage hashes) and N2/N4 fail by design: this is a new multi-relation corpus (no RB-predecessor
chain), and seen constructions are intentionally reused for the held-out RELATION family, which the
original single-relation audit flags as cross-split family reuse. Those are not leakage: entity
disjointness, fact disjointness, and exact-duplicate checks all pass independently.

## Holdout axes
- cue_predicate: concessive, relative (LOCAL structure, predicate never in training).
- structural: long_gap (long filler-gap), embedded (nested clauses) -- genuinely distinct structure.
- compositional: appositive on abbreviation (predicate seen via capital, relation seen via copular/
  possessive, the PAIRING unseen).
- relation: the entire currency_code family across constructions.

## Result (20 seeds, ENV1 CUDA)
- aggregate held-out exact-match: 99.3% [97.0%/99.9%], std 0.008; wrong-mapping 0.6%.
- per held-out construction (exact/wrong, min, std):
  - appositive 100.0%/0.0% (min 100.0%, std 0.000)
  - concessive 98.6%/1.4% (min 89.5%, std 0.024)
  - copular    100.0%/0.0% (min 100.0%, std 0.000)
  - relative   99.6%/0.0% (min 96.1%, std 0.011)
  - long_gap   100.0%/0.0% (min 99.7%, std 0.001)
  - embedded   98.8%/1.2% (min 94.8%, std 0.015)
- per axis: compositional 100.0%, cue_predicate 98.2%, structural 99.4%, relation 99.8%.
- per relation: capital 98.5%, abbreviation 99.6%, currency code (held-out) 99.8%.
- NO-MEMORY causal control (zeroed content): 51.6% exact; binder uplift +47.7%.
- second environment (CPU, 8 seeds): median 99.3%, delta vs CUDA 0.1%.

## Pilot (entity-disjoint calibration; thresholds frozen BEFORE the certification run)
relative per-seed min on calibration = 99.1% -> G_RELATIVE no-seed floor frozen at 94%;
compositional median = 100.0% -> G_COMPOSITIONAL floor frozen at 70%.
Certification met both: relative per-seed min 96.1% >= 94%; compositional 100.0% >= 70%.

## Gates (pre-declared / pilot-frozen) - all 11 PASS
- [PASS] G_GEN: held-out median exact 99.3% [97.0%/99.9%] (floor 75%); lexical 0.0%; uplift +99.3%.
- [PASS] G_PER_CONSTRUCTION: every held-out construction exact>=65% & wrong<=15%; failing=none; appositive:100%/0%; concessive:99%/1%; copular:100%/0%; embedded:99%/1%; long_gap:100%/0%; relative:100%/0%
- [PASS] G_RELATIVE: relative exact 99.6% wrong 0.0%; per-seed min 96.1% >= pilot floor 94%.
- [PASS] G_SAFE: aggregate wrong-mapping median 0.6% (ceiling 12%; RB3 31.4%).
- [PASS] G_CALIB: ambiguous abstain median 100.0% (floor 70%); known false-abstain median 0.0%.
- [PASS] G_STABILITY: min exact 97.0% (floor 65%; RB3 56.9%) & std 0.008 (ceiling 0.07) over 20 seeds.
- [PASS] G_COMPOSITIONAL: compositional (seen relation x seen predicate, unseen pairing) exact 100.0% >= pilot floor 70%; wrong 0.0%.
- [PASS] G_STRUCTURAL: held-out structural constructions verified distinct from training (long_gap/embedded gap+depth disjoint from local): True. Structural-construction performance: embedded:99%/1%; structural_fail=none.
- [PASS] G_NO_MEMORY: binder median exact 99.3% vs zeroed-content control 51.6%, gap +47.7% (need >= 20%).
- [PASS] G_REPRO: second env (CPU, 8 seeds) median exact 99.3% vs ENV1 99.3%, delta 0.1% (tol 5%); same machine, CPU vs CUDA backend (NOT distinct hardware).
- [PASS] G_SEALS: sealed artifacts byte-identical before=True after=True; substrate SHA 8eb5362ed39fd6b9 read-only.

## Failure taxonomy (representative median seed)
wrong_binding (identity<->swapped confusion) ~9.0 items; false_abstain 0.0;
missed_abstain 0.0%; malformed 0. Most failures concentrate on
concessive (4.1% wrong). A 3-way head
(identity/swapped/abstain) cannot separate wrong-entity from wrong-value; wrong_binding is the
identity/swapped confusion rate. reports/seed_variance_diagnosis.md localizes the residual variance
to the concessive construction (std 0.024); no pipeline instability dominates.

## Honest mechanistic interpretation (do not overread)
The binder is POSITION-BLIND: it pools the entity and value spans by content and matches them, with
one relation embedding. The structural constructions (long filler-gap, clause embedding) change
surface structure but do NOT break the binder, because (a) it does not use position, and (b) the
frozen substrate still encodes the entity->value binding via attention across the gap/embedding. The
no-memory control (51.6%) proves the binding signal is read from substrate content, not
from position or structure. Therefore the structural result demonstrates ROBUSTNESS of content-
addressed binding to structural variation (and of the substrate attention under structural stress),
NOT syntactic parsing by the binder. This is the correct, scoped reading.

## Claim status (claim ladder)
All 11 gates pass at 20 seeds in two backends, per-construction floors held (incl. the
structural and relative constructions), G_STRUCTURAL distinctness verified, seals byte-identical.
Per the claim ladder this supports: D_Cortex ContentAddressedRoleBinder demonstrates systematic
generalization to unseen, structurally distinct cue constructions (long filler-gap, clause
embedding), including relative structure, with calibrated abstention and a 47.7pp causal margin over
a no-memory control, under non-lexical RB4 conditions -- with the mechanistic scope above.
SCOPE / NOT PROVEN: still simple, explicitly-cued bindings; the second environment is CPU vs CUDA on
the SAME machine (not distinct hardware); single organization. PROVEN for patent/EESR still requires
true multi-hardware reproduction and independent replication. Do not promote this MEASURED result to
PROVEN.
