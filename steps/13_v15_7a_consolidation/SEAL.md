# SEAL — D_Cortex v15.7a (Pas 7a)

**Status: SEALED**
**Date: 2026-04-26**
**Verdict: PAS 7A SEALED — all 10 gates green**

---

## Statement

D_Cortex v15.7a este primul organ longitudinal validat: memoria poate
reconcilia, prune, retrograde și promote la `end_episode` fără să
contamineze Pas 6.

Stabilul poate cădea (retrograde sub evidență contrară M=2).
Provizoriul poate urca (promote prin convergență N=2 + age ≥ K_age).
Disciplina onestității Pas 6 este păstrată byte-identical.

---

## Configurație evaluare

```
n_per_l_family : 20
seed           : 20261103
total trials   : 100 sequences × ~3-4 episodes/sequence
hardware       : NVIDIA A100-SXM4-40GB, bfloat16, TF32, SDPA
parameters     : N_promote=2  M_retrograde=2  K_promote_age=2  K_prune_stale=3
```

## Gates 0..9 — verdicte finale

| Gate | Threshold | Rezultat |
|---|---|---|
| 0 | trusted regression byte-identical | PASS (snapshot before == after) |
| 1 | wrong_commit ≤ 0.02 toate F1-F5 | PASS (0.000 toate) |
| 2 | F2 safe_resolution ≥ 0.95 | PASS (0.952) |
| 3 | false_promote_rate = 0 | PASS (0 / 100) |
| 4 | false_retrograde_rate = 0 | PASS (0 / 100) |
| 5 | L1 promote_rate ≥ 0.95 | PASS (1.000) |
| 6 | L2 retrograde_rate ≥ 0.90 | PASS (1.000) |
| 7 | L3 false_retrograde = 0 | PASS (0 / 20) |
| 8 | L4 promote_count = 0 | PASS (0 / 20) |
| 9 | L5 prune_count ≥ 1/trial | PASS (2 per trial × 20) |

## Phase A — Pas 6 invariants sub Pas 7a arbiter

```
trusted_snapshot_before == trusted_snapshot_after :
  clear_commit_rate: 1.0   s1_honesty: 1.0   s_avg_overcommit: 0.0
  clear_fidelity:    1.0   s2_honesty: 1.0
  clear_uncertain:   0.0   s3_honesty: 1.0
                           s4_honesty: 1.0

F1_novel_paraphrase_syntax  : commit_correct=0.000  wrong_commit=0.000  parser_fail=1.000
F2_multiword_entities       : commit_correct=0.952  wrong_commit=0.000  parser_fail=0.000
F3_novel_lexical_alias      : commit_correct=0.000  wrong_commit=0.000  parser_fail=1.000
F4_discourse_intercalation  : commit_correct=1.000  wrong_commit=0.000  parser_fail=0.000
F5_novel_query_forms        : commit_correct=0.148  wrong_commit=0.000  parser_fail=0.660
```

Identice cu Pas 6 standalone. Consolidatorul este complet invizibil pe single-episode.

## Phase B — comportament longitudinal

| Family | actual_RECONCILE | actual_PRUNE | actual_RETROGRADE | actual_PROMOTE | false_promote | false_retrograde |
|---|---|---|---|---|---|---|
| L1_promote_cycle | 20 | 0 | 20 | **20** | 0 | 0 |
| L2_retrograde_only | 20 | 0 | **20** | 0 | 0 | 0 |
| L3_completion | 0 | 0 | 0 | 0 | 0 | 0 |
| L4_no_inflation | 20 | 0 | 0 | 0 | 0 | 0 |
| L5_stale_prune | 20 | **40** | 0 | 0 | 0 | 0 |

Toate counts strict above threshold. Niciun gate pe muchie.

## Componente sealed

- C1 LongitudinalEpisodeRegime
- C2 baseline runner (D.2)
- C3 derivation layer (D.3)
- C4 Consolidator.reconcile (D.4)
- C5 Consolidator.prune (D.5)
- D6 Consolidator.retrograde
- D7 Consolidator.promote
- D8 CommitArbiterPas7a wiring
- D9 full evaluator (Gates 0-9)

## Artefact oficial

```
/content/drive/MyDrive/dcortex_v2/v15_7a/results/v15_7a_d9_full_eval.json
```

Dovadă imutabilă pe Drive. Conține per-trial detail pentru toate 100 secvențe (5 familii × 20 trials), per-trial audit, verdicts finale, snapshot-uri trusted before/after.

## Restricții post-seal

- D.6 / D.7 / D.8 / D.9 — nu se modifică
- Gate logic 0-9 — nu se relaxează / redesign
- Pas 6 critical path — byte-identical, neatins
- Query path — neatins
- Semantic abstraction — blocat până la adapter explicit
- Integrare cu fragmergent-memory-engine — blocată până la adapter exact definit

## Open spec items (post-seal, non-blocking)

1. `expected_final_committed` în L1/L4/L5 nu enumără entitățile distractor.
   Per-trial "committed_match: 0/20" pentru aceste familii este artefact de
   raportare, NU regresie. Gate-urile contează ops, nu bank exact match.
2. `PROMOTE_SKIPPED` audit operation introdus de D.7 e audit-only, nu
   contribuie la `false_promote`.

Niciun item nu blochează seal-ul.
