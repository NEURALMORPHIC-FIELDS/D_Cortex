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

---

## [14] v15.7b-Q — Contextual Semantic Query Internalizer

- **Folder**: [`steps/14_v15_7b_semantic_adapter/`](../steps/14_v15_7b_semantic_adapter/)
- **Data**: 2026-06-15, verdict `2026-06-15T18:57:17+03:00`
- **Scop**: adapter explicit și internalizator semantic query-side conservator,
  fără acces la mutația memoriei Pas 7a.
- **Evoluție măsurată**:
  1. token-mean producer: honest, dar `0/20` emissions pe novel forms
  2. likelihood single-view: `19/32`, ambiguity `15/16`
  3. likelihood multi-view F5: `317/500`, wrong `3.6%`
  4. trained standard-only: F5 `87.6%`, dar F1/F3 `47.2%/50.6%`
  5. pooled leave-one-form-out: F3 `87.7%`, F5 `94.2%`, F1 `69.9%`
  6. contextual leave-one-form-out: toate cele 11 gates PASS
- **Metrici finale query-side**:
  - F1 contextual out-of-fold: `451/528 = 85.4%`
  - F3 contextual out-of-fold: `474/528 = 89.8%`
  - F5 contextual final: `463/500 = 92.6%`
  - wrong emitted: `0.0% / 0.0% / 0.2%`
  - ambiguous abstention: `200/200`
- **Invariante**:
  - toate emissions trec prin adapter și sunt `QUERY_ONLY`
  - zero memory read/write în contextual feature path
  - substrate byte-identical, zero trainable substrate parameters
  - Pas 7a SHA-256 neschimbat
- **Verdict**: **QUERY-SIDE SUBMILESTONE SEALED**
- **Limită explicită**: nu este fact-side internalization și nu este evaluare
  end-to-end F1/F3/F5 asupra commit-ului în memorie.
- **Seal**: [`steps/14_v15_7b_semantic_adapter/SEAL.md`](../steps/14_v15_7b_semantic_adapter/SEAL.md)
- **Succesor permis**: bridge read-only explicit către query routing, urmat de
  evaluare end-to-end înghețată; nicio scriere directă în committed memory.

---

## [15] v15.7b-R — Read-Only Semantic Query Bridge

- **Folder**: [`steps/15_v15_7b_read_only_bridge/`](../steps/15_v15_7b_read_only_bridge/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:12:02+03:00`
- **Scop**: conectarea interpretării semantice query-side la o rută canonică
  de citire, fără nicio capabilitate de mutație.
- **Contract bridge**: PASS — accepted-query-only, fallback exact,
  determinism, zero mutation API, stări memory byte-identical.
- **Calitate semantic route**:
  - F1 `88.5%`, wrong `0.0%`
  - F3 `93.5%`, wrong `0.0%`
  - F5 `94.5%`, wrong `0.0%`
- **Verdict end-to-end neural-memory**: **FAIL**
  - F1 recall `13.0% -> 13.5%`
  - F3 recall `36.0% -> 36.0%`
  - F5 recall `36.5% -> 36.5%`
  - S5/S6 honesty `0.0% / 0.0%`, overcommit `100% / 100%`
- **Interpretare**: query semantics există, dar nu repară scrierea/value
  emission și nu poate adăuga onestitate unui reader care emite obligatoriu.
- **Scope guard**: nu este integrare Pas 7a și nu este avantaj semantic-memory.
- **Verdict**:
  [`steps/15_v15_7b_read_only_bridge/VERDICT.md`](../steps/15_v15_7b_read_only_bridge/VERDICT.md)
- **Succesor justificat**: fact-side semantic hypotheses strict
  `PROVISIONAL_ONLY`; nu încă o iterație de query routing.

---

## [16] v15.7b-F — Fact-Side Provisional Semantic Producer

- **Folder**: [`steps/16_v15_7b_fact_provisional/`](../steps/16_v15_7b_fact_provisional/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:22:25+03:00`
- **Scop**: interpretare contextuală `(entity, attribute, value)` pentru fapte
  F1, exclusiv prin adapter `PROVISIONAL_ONLY`.
- **Verdict**: **HONEST PARTIAL**
  - out-of-fold accuracy `1199/2000 = 60.0%` — FAIL
  - wrong provisional `22/2000 = 1.1%` — PASS
  - ambiguity abstention `2400/2400 = 100%` — PASS
  - emissions adapter provisional-only `1221/1221` — PASS
- **Limită măsurată**: head-ul global `attribute:value` produce predominant
  mismatch/abstention după ce atributul fusese deja clasificat separat.
- **Scope guard**: zero ingestion/promote în Pas 7a, zero committed write.
- **Verdict**:
  [`steps/16_v15_7b_fact_provisional/VERDICT.md`](../steps/16_v15_7b_fact_provisional/VERDICT.md)
- **Succesor justificat**: attribute-conditioned value decoding, cu
  `UNKNOWN_VALUE` și marginile neschimbate, pe eșantion nou înghețat.

---

## [17] v15.7b-F2 — Attribute-Conditioned Fact Decoder

- **Folder**: [`steps/17_v15_7b_conditioned_fact/`](../steps/17_v15_7b_conditioned_fact/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:28:26+03:00`
- **Scop**: testarea ipotezei că mismatch-ul global `attribute:value` era
  cauza principală a acoperirii fact-side reduse.
- **Verdict**: **NEGATIVE; BRANCH STOPPED**
  - conditioned accuracy `1211/2000 = 60.6%` — FAIL
  - wrong provisional `76/2000 = 3.8%` — FAIL
  - uplift față de același predecessor `+0.2pp` — FAIL
  - ambiguity abstention `2400/2400 = 100%` — PASS
- **Interpretare**: filtrarea post-hoc nu creează evidență de role binding;
  amplifică uneori o valoare slabă din atributul acceptat și produce emisii
  greșite.
- **Scope guard**: zero Pas 7a ingestion/commit; substrate și seal-uri
  neschimbate.
- **Verdict**:
  [`steps/17_v15_7b_conditioned_fact/VERDICT.md`](../steps/17_v15_7b_conditioned_fact/VERDICT.md)
- **Decizie**: fără alte filtre post-hoc și fără relaxarea pragurilor.

---

## [18] v15.7b-O — Direct Semantic Object Read

- **Folder**: [`steps/18_v15_7b_semantic_object_read/`](../steps/18_v15_7b_semantic_object_read/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:39:32+03:00`
- **Scop**: citire directă a unui snapshot obiectual epistemic din coordonata
  semantică aprobată, fără reconversie în text și fără parser.
- **Verdict**: **HONEST PARTIAL**
  - F1/F3/F5 direct correct `84.5% / 91.0% / 93.5%`
  - wrong committed read `0.0% / 0.0% / 0.0%`
  - S5 honesty `100%`, overcommit `0%`
  - S6 honesty `93.5%`, overcommit `6.5%`
- **Interpretare**: memoria obiectuală răspunde sigur unei coordonate corecte,
  dar internalizatorul poate aproba un referent arbitrar pentru query-uri
  pronume-only.
- **Limită explicită**: nu este integrare runtime Pas 7a; snapshot-urile
  known-family sunt prepopulate, deci fact-side rămâne nemăsurat.
- **Verdict**:
  [`steps/18_v15_7b_semantic_object_read/VERDICT.md`](../steps/18_v15_7b_semantic_object_read/VERDICT.md)
- **Succesor justificat**: guard conservator de referent explicit; F1 coverage
  rămâne deschis.

---

## [19] v15.7b-G — Explicit Referent Grounding

- **Folder**: [`steps/19_v15_7b_referent_grounding/`](../steps/19_v15_7b_referent_grounding/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:44:57+03:00`
- **Scop**: blocarea coordonatelor semantic-query fără dovadă explicită a
  referentului în source text.
- **Rezultat absolut**:
  - S6 overcommit `3.5% -> 0.0%`
  - S6 honesty `96.5% -> 100%`
  - F1/F3/F5 și S5 fără regresie
- **Verdict frozen**: **8 PASS, 1 FAIL**
  - H3 uplift a cerut `+5pp`; măsurat `+3.5pp`
- **Decizie**: pragul nu este relaxat și seed-ul nu este rerulat. Guardul
  rămâne safety improvement verificat, dar milestone-ul all-gates nu este
  sigilat.
- **Verdict**:
  [`steps/19_v15_7b_referent_grounding/VERDICT.md`](../steps/19_v15_7b_referent_grounding/VERDICT.md)
- **Succesor justificat**: benchmark fact-side de role binding ne-rezolvabil
  prin token overlap, înainte de alt model.

---

## [20] v15.7b-RB0 — Non-Trivial Role-Binding Benchmark

- **Folder**: [`steps/20_v15_7b_role_binding_benchmark/`](../steps/20_v15_7b_role_binding_benchmark/)
- **Data**: 2026-06-15, verdict `2026-06-15T19:49:55+03:00`
- **Verdict**: **8/8 PASS; BENCHMARK SEALED**
- **Rezultat**:
  - best lexical/position baseline `37.1%` exact-known
  - ordered baseline `25.0%`
  - lexical Cartesian `0.0%`
  - safe abstain `0%` coverage
  - toate baseline-urile non-abstaining: `100%` overcommit pe ambiguu
- **Interpretare**: inventarul lexical este disponibil, dar nu determină
  legarea entitate-valoare. Shortcut-ul lexical măsurat inițial este eliminat
  în acest benchmark controlat.
- **Verdict**:
  [`steps/20_v15_7b_role_binding_benchmark/VERDICT.md`](../steps/20_v15_7b_role_binding_benchmark/VERDICT.md)
- **Succesor justificat**: un singur experiment de learned role binder,
  strict provisional-only.

---

## [21] v15.7b-RB1 — Conservative Learned Role Binder

- **Folder**: [`steps/21_v15_7b_role_binder/`](../steps/21_v15_7b_role_binder/)
- **Data**: 2026-06-15, verdict `2026-06-15T20:03:22+03:00`
- **Verdict frozen după corecția strictă J10**: **8 PASS, 3 FAIL**
- **Rezultat**:
  - validation loss drop `46.2%`
  - known test exact binding `124/256 = 48.4%`
  - wrong emitted mapping `132/256 = 51.6%`
  - ambiguity abstention `57/57 = 100%`
  - uplift peste best same-test lexical `+11.7pp`
- **Interpretare**: head-ul pooled învață clasa unresolved, dar nu distinge
  relațional identity de swapped; toate familiile cunoscute rămân la
  aproximativ `41–52%`.
- **Corecție harness**: J10 inițial fals-negativ dintr-un literal SHA-256 cu un
  caracter omis; numai metadatele J10 au fost corectate, fără reantrenare,
  recalibrare sau rerulare de test.
- **Verdict**:
  [`steps/21_v15_7b_role_binder/VERDICT.md`](../steps/21_v15_7b_role_binder/VERDICT.md)
- **Decizie**: ramura candidate-view pooling este oprită.
- **Succesor justificat**: token-level role-conditioned sequence scorer, fără
  relation lexicon și strict provisional-only.

---

## [22] v15.7b-RB2 — Token-Level Role-Conditioned Binder

- **Folder**:
  [`steps/22_v15_7b_role_conditioned_binding/`](../steps/22_v15_7b_role_conditioned_binding/)
- **Data**: 2026-06-15, verdict `2026-06-15T20:14:04+03:00`
- **Verdict**: **13/13 PASS; CONTROLLED SUBMILESTONE SEALED**
- **Rezultat**:
  - complete role masks `2000/2000`
  - validation loss drop `99.1%`
  - known test exact binding `256/256 = 100%`
  - wrong emitted mapping `0/256 = 0%`
  - ambiguity abstention `57/57 = 100%`
  - uplift față de RB1 `+51.6pp`
  - uplift față de best same-test lexical `+63.3pp`
- **Interpretare**: candidate-role masks peste stările contextuale token-level
  elimină simetria identity/swapped observată în RB1.
- **Scope guard**: toate familiile sintactice cunoscute apar în training; nu
  este încă dovadă unseen-syntax sau integrare de memorie.
- **Verdict**:
  [`steps/22_v15_7b_role_conditioned_binding/VERDICT.md`](../steps/22_v15_7b_role_conditioned_binding/VERDICT.md)
- **Succesor justificat**: leave-one-syntax-family-out cu arhitectura și
  pragurile RB2 neschimbate, înainte de orice ingestion.

---

## [23] v15.7b-RB3 — Leave-One-Syntax-Family-Out

- **Folder**:
  [`steps/23_v15_7b_syntax_holdout/`](../steps/23_v15_7b_syntax_holdout/)
- **Data**: 2026-06-15, verdict `2026-06-15T20:20:25+03:00`
- **Verdict**: **9 PASS, 3 FAIL; MEMORY INTEGRATION BLOCKED**
- **Rezultat**:
  - aggregate unseen-syntax exact `911/1600 = 56.9%`
  - aggregate wrong mapping `503/1600 = 31.4%`
  - minimum family exact `39.2%`
  - ambiguity abstention `100%` în fiecare fold
  - uplift față de best aggregate lexical `+19.9pp`
- **Per familie**: RB1 `62.8%`, RB2 `39.2%`, RB3 `79.3%`, RB4 `46.5%`.
- **Interpretare**: RB2 rezolvă seen-syntax, dar nu generalizează suficient de
  sigur la construcții nevăzute. Optimizerul și măștile nu sunt bottleneck-ul.
- **Verdict**:
  [`steps/23_v15_7b_syntax_holdout/VERDICT.md`](../steps/23_v15_7b_syntax_holdout/VERDICT.md)
- **Decizie**: fără ingestion și fără alt tuning pe aceleași patru familii.
- **Succesor justificat**: poartă data-only pentru un corpus independent,
  auditat și separat pe familii de construcție.
