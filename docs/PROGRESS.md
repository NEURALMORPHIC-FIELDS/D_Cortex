# PROGRESS.md — Jurnal succesiv de dezvoltare

Fiecare intrare = un pas sigilat. Ordine cronologică. Nu se rescrie istoria; dacă un pas a avut o greșeală, corecția apare în pasul următor și e menționată aici.

Format per intrare:
- **Step**: folder + numele logic
- **Data**: când s-a încheiat
- **Scop**: ce problemă atacă
- **Schimbări față de predecesor**: delta esențială
- **Verdict**: PASS / PARTIAL / FAIL
- **Metrici cheie**: numere, nu impresii
- **Scope guard**: ce NU a fost atins
- **Predecessor** / **Succesor**

---

## [08-10] v15.5 → v15.6 Pas 3 (bundle istoric)

- **Folder**: [`steps/08-10_v15_5_to_v15_6_pas3/`](steps/08-10_v15_5_to_v15_6_pas3/)
- **Data**: pre-2026-04
- **Scop**: de la external holdout robustness (v15.5) până la EntitySpanComposer pentru F2 (v15.6 Pas 3).
- **Natură**: `code.py` este un concat al trei step-uri Colab (step8 + step9 + step10), exportat într-un moment în care nu exista încă disciplina de versionare per step. Rămâne sigilat ca atare, ca baseline istoric.
- **Verdict global**: PARTIAL — v15.6 Pas 3 a ridicat F2 safe_resolution la ~0.78 prin EntitySpanComposer, dar F2 uncertain rămânea la 21.8% (attr_write_failure). Diagnostic-ul Pas 3.1a a falsificat ipoteza „simetrie write/read" și a redirecționat atacul spre Pas 6 (RoMR).
- **Ce conține intern**:
  - v15.5: external holdout generator (5 familii + S5/S6), evaluator cu 10 gates
  - v15.6 Pas 1: InternalizationPacket + structural wrapper (echivalență byte-identică cu v15.4.1)
  - v15.6 Pas 2: CommitArbiter + ProvisionalMemory + EpisodeBuffer (zona provizorie + arbitraj commit)
  - v15.6 Pas 3: EntitySpanComposer (rule-based, max 2 modifiers, conservator)
  - v15.6 Pas 3.1a: F2 causal diagnosis offline (FALSIFICAT → Pas 6 justificat)
- **Predecessor**: —
- **Succesor**: `12_v15_6_pas6_romr`

---

## [12] v15.6 Pas 6 — Role-of-Modifier Resolver (RoMR)

- **Folder**: [`steps/12_v15_6_pas6_romr/`](steps/12_v15_6_pas6_romr/)
- **Data**: 2026-04-22 (rulare A100)
- **Scop**: închiderea cauzei reale a F2 uncertain — sistemul confunda ENTITY_MODIFIER cu ATTRIBUTE_VALUE înainte de commit.
- **Schimbări față de predecesor**:
  1. Nou: `RoleOfModifierResolver` — clasifică fiecare `value_candidate` ca ENTITY_MODIFIER / ATTRIBUTE_VALUE / UNCERTAIN, pe baza poziției față de NP span și copula.
  2. Nou: packet-level `REAL_CONFLICT` — promovat la ATTR_CONFLICT_STRONG când aceeași familie atributivă are ≥ 2 valori distincte (cazuri „The small horse is huge").
  3. Nou: recompute flag după filtrare — flags value-dependent (MULTIPLE_ATTR_TRIGGERS, ATTR_CONFLICT_STRONG, ATTR_VALUE_MISMATCH, VALUE_MISSING_OR_UNCLEAR) sunt re-derivate; cele independente (TEMPLATE_UNKNOWN, REFERENT_AMBIGUOUS, etc.) sunt păstrate.
  4. Nou: filtrare coerentă a attribute_candidates cu filtered value_candidates (evită ca v15.4 să aleagă o familie fără valoare suport).
  5. Integrare: `CommitArbiterPas6` extinde `CommitArbiterPas3`; RoMR rulează DUPĂ v15.4 parser, ÎNAINTE de verifier, pe shallow copy al packet-ului. Raw packet păstrat integral pentru audit.
- **Verdict**: **PAS 6 PASSED pe A100** — toate 7 gates verzi.
- **Metrici cheie (F2)**:

  | Metric | Pas 3 baseline | Pas 6 A100 |
  |---|---:|---:|
  | safe_resolution | 0.782 | **0.952** |
  | uncertain | 0.218 | 0.048 |
  | wrong_commit | 0.000 | 0.000 |
  | attr_write_fail post-RoMR | 0.218 | **0.000** |

- **Metrici globale**:
  - Trusted regression: byte-identical before/after
  - S5/S6 honesty: 1.000 / overcommit: 0.000
  - F4 safe_resolution: 1.000
  - REAL_CONFLICT detectat: 24/500 trials F2
  - ENTITY_MODIFIER tokens dropped: 85
- **Scope guard** (explicit NE-atins, conform directivei):
  - query path (RoMR e fact-only)
  - parser v15.4 general
  - substrate v15.1
  - shadow checkpoint (frozen oracle)
  - Pas 2 arbiters în afara integrării RoMR (CommitArbiterPas6 extinde, nu modifică)
  - F1/F3/F5 — rămân la metrici identice cu Pas 3 (commit_correct 0.000/0.000/0.148). Tratate informațional, nu sunt target Pas 6.
- **Checkpoint**: `/content/drive/MyDrive/dcortex_v2/v15_6/results/v15_6_pas6_romr.json`
- **Predecessor**: `08-10_v15_5_to_v15_6_pas3` (bundle)
- **Succesor**: `13_...` (în decizie — probabil strat de consolidare / replay, vezi secțiunea Next)

---

## [13] v15.7a — Consolidator sincron la end_episode

- **Folder**: [`steps/13_v15_7a_consolidation/`](steps/13_v15_7a_consolidation/)
- **Data**: 2026-04-26 (rulare A100, validat empiric)
- **Scop**: primul mecanism prin care memoria operează asupra propriei istorii — reconcile, prune, retrograde, promote la `end_episode`, sincron, după finalize-ul Pas 2/6. Definește dinamica longitudinală reală (stabilul poate cădea, provizoriul poate urca) fără a contamina critical path-ul Pas 6.
- **Schimbări față de predecesor**:
  1. Nou: `LongitudinalEpisodeRegime` (C1) — 5 generatori L1-L5 pentru secvențe cross-episode (promote_cycle, retrograde_only, completion, no_inflation, stale_prune).
  2. Nou: `ProvisionalEntry` derivation layer (C3) — predicate pure peste provisional: `_v15_7a_confirmation_episodes`, `_last_activity_episode`, `_first_seen_episode`, `_distinct_values_for_slot`; predicate operationale `is_promote_eligible`, `is_retrograde_eligible`, `is_stale_for_prune`.
  3. Nou: `Consolidator.reconcile` (C4) — colapsează duplicate (slot, value, episode_id) în provisional, nu atinge bank.
  4. Nou: `Consolidator.prune` (C5) — drop entries când slot e stale (K=3 episoade fără activitate); per-entry counting.
  5. Nou: `Consolidator.retrograde` (D6) — prima operație care MUTEAZĂ bank-ul. Demote committed slot când M=2 challenger episodes distincte; mutație in-place pe AttributeSlot, șterge stability_index entry, NU re-adaugă valoarea retrogradată în provisional în v1.
  6. Nou: `Consolidator.promote` (D7) — atomic, intra-pas exclusion via audit-scan (slot retrogradat în acest end_episode e blocat); bank-state policy: empty → promote, same-value → idempotent finalize, different-value-stable → PROMOTE_SKIPPED (no transitive demote v1); curăță provisional (slot, promoted_value).
  7. Nou: `CommitArbiterPas7a(CommitArbiterPas6)` (D8) — override pe `end_episode` care rulează `_v15_7a_run_consolidator_pipeline` (reconcile → prune → retrograde → promote) după Pas 2/6 finalize. In-episode (write_fact, RoMR, dual conflict rule, cross-episode challenger) complet neatins.
  8. Nou: `v15_7a_run_full_eval_d9` (D9) — evaluator end-to-end peste 10 gates. Phase A re-rulează F1-F5 + S5/S6 cu Pas 7a arbiter. Phase B rulează L1-L5 cu n=20/familie. Salvează raport JSON.
- **Verdict**: **PAS 7A SEALED** — all 10 gates green pe A100.
- **Metrici cheie**:

  | Gate | Threshold | Rezultat |
  |---|---|---:|
  | 0 | trusted regression byte-identical | PASS |
  | 1 | wrong_commit ≤ 0.02 toate F1-F5 | 0.000 toate |
  | 2 | F2 safe_resolution ≥ 0.95 | 0.952 |
  | 3 | false_promote_rate = 0 | 0 / 100 |
  | 4 | false_retrograde_rate = 0 | 0 / 100 |
  | 5 | L1 promote_rate ≥ 0.95 | 1.000 (20/20) |
  | 6 | L2 retrograde_rate ≥ 0.90 | 1.000 (20/20) |
  | 7 | L3 false_retrograde = 0 | 0 / 20 |
  | 8 | L4 promote_count = 0 | 0 / 20 |
  | 9 | L5 prune_count ≥ 1/trial | 2 per trial × 20 |

- **Metrici globale**:
  - Pas 6 invariants byte-identical sub Pas 7a arbiter (F2 0.952, S5/S6 honesty 1.000, F4 1.000)
  - Total ops Phase B: RECONCILE=80, PRUNE=40, RETROGRADE=40, PROMOTE=20
  - n_per_l_family=20, seed=20261103, 100 secvențe × ~3-4 episodes/seq
- **Patch-uri minore aplicate post-D9** (ne-funcționale pentru critical path):
  1. L2 ep3 template: `"A {chall_val} {entity} stood nearby."` → `"The {entity} stood {chall_val} nearby."` (post-copular, RoMR clasifică ATTRIBUTE_VALUE). Cauza primară pentru Gate 6 0/20 inițial — RoMR Pas 6 filtra blue ca ENTITY_MODIFIER în NP interior. Post-patch: 20/20 retrogrades.
  2. JSON serializer: `_v15_7a_json_safe()` convertește chei tuple `(entity_id, attr_type)` în string `"entity::attr"`. Strict serializer-only.
- **Scope guard** (explicit NE-atins):
  - Pas 6 critical path (byte-identical confirmat)
  - query path (RoMR fact-only, Pas 7a la end_episode)
  - substrate v15.1
  - shadow checkpoint (frozen oracle)
  - parser v15.4 general
  - F1/F3/F5 — rămân la metrici identice cu Pas 6 (informațional)
  - semantic abstraction — blocat până la adapter explicit
  - integrare cu fragmergent-memory-engine — blocată până la adapter exact definit
- **Artefact oficial**: `/content/drive/MyDrive/dcortex_v2/v15_7a/results/v15_7a_d9_full_eval.json`
- **Seal**: [`steps/13_v15_7a_consolidation/SEAL.md`](steps/13_v15_7a_consolidation/SEAL.md)
- **Predecessor**: `12_v15_6_pas6_romr`
- **Succesor**: nedecis (vezi Next)

---

## Next (nedecis, nu-i step sigilat încă)

D_Cortex v15.7a este acum primul organ longitudinal validat. Memoria operează asupra propriei istorii la `end_episode`. Stabilul poate cădea, provizoriul poate urca, fără să contamineze Pas 6.

Două direcții deschise (nu se începe niciuna fără adapter explicit):

- **7b (semantic abstraction conservator)**: strat care produce ipoteze pe care consolidatorul le poate metaboliza — nu sinonime hardcodate. Targetul: F1/F3/F5 (parafraze, alias lexical, forme novel de query) care rămân la 0.000-0.148 commit_correct sub Pas 6+7a. Cu consolidatorul live, ipotezele semantice pot fi tratate ca provisional și convergența poate determina commitment, nu o regulă rigidă.
- **8 (integrare cu fragmergent-memory-engine)**: D_Cortex 7a ca backend latent longitudinal la end_episode al organismului explicit. Cere adapter exact definit (interfața consolidator + audit log + provenance) înainte de orice atingere a codului.

Regula firmă: niciuna dintre direcții nu se începe fără un adapter scris explicit. Pas 7a stă sigilat ca atare.
