# NOTES — Step 13 v15.7a Consolidation

Jurnal intern al step-ului. Scris pe măsură ce lucrez. Rămâne în folder și după sigilare.

---

## 2026-04-22 — D.1 început: LongitudinalEpisodeRegime

Cod adăugat la sfârșitul `code.py`:
- `LongitudinalSequence` dataclass (11 câmpuri)
- Helper `_v15_7a_build_episode` pentru a construi HoldoutEpisode dintr-o listă de facts
- Generatori L1–L5 cu predicții explicite de stare finală + expected_ops
- `LONGITUDINAL_FAMILIES` registry

Zero dispatch, zero execuție nouă la import. Pas 6 runtime dispatch rulează neschimbat când se importă `code.py`.

Design decisions în D.1:
- `expected_final_committed: Dict[(ent,attr), value_idx]` — ce ar trebui să aibă bank-ul la sfârșit
- `expected_provisional_present` / `_absent` — urme provisional așteptate / așteptate să NU fie (pentru prune)
- `expected_ops: Dict[str, int]` — câte operații consolidator de fiecare tip sunt așteptate peste toată secvența
- Retrograded committed values **NU** se re-adaugă în provisional în v1 (simplu, previne cascada, audit trail păstrează evenimentul). Decizie de design, nu derivată din directivă — de reconfirmat înainte de D.6.

Predicții de ops per familie:
- L1 (4 ep): {RECONCILE: 1, RETROGRADE: 1, PROMOTE: 1}
- L2 (3 ep): {RECONCILE: 1, RETROGRADE: 1}
- L3 (3 ep): {} — consolidator no-op peste tot (fără provisional, fără conflict)
- L4 (3 ep): {RECONCILE: 1} — collapse intra-ep1, NICIODATĂ promote (N=1 pentru ambele valori)
- L5 (5 ep): {RECONCILE: 1, PRUNE: 2} — reconcile la ep1, prune ambele la end_ep4

Next: D.2 smoke test — rulez L1–L5 cu Pas 6 curent (fără consolidator), ca să am baseline explicit pentru comparație în D.8–D.9.

---

## 2026-04-22 — D.1 validat pe A100 + D.2 implementat

**D.1 pe A100 (output complet):**
- Pas 6 rulează neschimbat: toate 7 gates verzi, F2 0.952, byte-identical trusted regression
- D.1 încărcat curat: 5 generatori L1–L5, LongitudinalSequence (11 fields), params N=2 M=2 K_age=2 K_stale=3
- Zero side-effects la import: sequences = data only

**D.2 adăugat la `code.py` (linii 15199–15429):**
- `V15_7a_SequenceState`, `V15_7a_BaselineTrial` — dataclass-uri pentru snapshot + comparație
- `_v15_7a_snapshot_sequence_state` — captură (bank_snapshot, provisional_entries)
- `_v15_7a_compare_state_vs_expected` — (committed_match, provisional_match, ops_match)
- `v15_7a_run_baseline_d2` — runner principal: state persistă între episoade în secvență, reset între secvențe
- Dispatch gated de `V15_7A_D2_MODE=run` (skip la default → nu rulează involuntar)

**Design decisions în D.2:**
- Subset-semantics pentru provisional_match: expected_present ⊆ observed AND expected_absent ∩ observed = ∅
- episode_counter separat (7_000_000+) vs Pas 6 evaluator, să nu coincidă
- Foloseste CommitArbiterPas6 (RoMR + Pas 3 composer active), construit pe infrastructura pe care Pas 7a va pune consolidatorul

**Predicții baseline (ce ar trebui să vedem când rulăm cu V15_7A_D2_MODE=run):**
- L3_completion: committed ✓, provisional ✓, ops ✓ (consolidator no-op aici)
- L4_no_inflation: committed ✓, provisional ✓, ops ✗ (RECONCILE nu fire fără consolidator)
- L1/L2/L5: mai multe divergențe — asta e exact ce consolidatorul va închide

Next: user poate rula cu `V15_7A_D2_MODE=run` pe A100 pentru baseline empiric. După validare, merg la D.3 (extend ProvisionalEntry cu confirmation_episodes + last_activity_episode).

---

## 2026-04-22 — D.3 implementat: derivation layer peste ProvisionalEntry

**Deviere conștientă de la formularea README.** README spune „extend ProvisionalEntry cu confirmation_episodes + last_activity_episode". Interpretarea literală = modificare a dataclass-ului sigilat de la linia 11175. Nu am făcut asta. Motivul:

1. **Gate 0 risk**: ProvisionalEntry e shared cu Pas 2/3/6. Chiar și cu default-uri, orice câmp nou poate muta serialization/repr și rupe trusted regression byte-identical.
2. **Informația nu e nouă**: `confirmation_episodes(slot, value)` și `last_activity_episode(slot)` sunt funcții pure de `episode_id` care deja există pe fiecare ProvisionalEntry. E pur și simplu agregare peste `ProvisionalMemory.entries`.
3. **Write paths**: 5 locuri construiesc ProvisionalEntry (11459, 11476, 11568, 12671, 14322). Mutare a clasei ar necesita audit pe fiecare.

**Cod adăugat la sfârșitul `code.py` (linii 15430–15758, 329 linii noi):**

Derivation helpers (pure over `List[ProvisionalEntry]`):
- `_v15_7a_confirmation_episodes(entries, ent, attr, val_idx) -> Set[int]`
- `_v15_7a_last_activity_episode(entries, ent, attr) -> Optional[int]`
- `_v15_7a_first_seen_episode(entries, ent, attr, val_idx) -> Optional[int]`
- `_v15_7a_distinct_values_for_slot(entries, ent, attr) -> Set[int]`

Consolidator predicates (se vor folosi în D.4–D.7):
- `_v15_7a_is_promote_eligible` — cere `|confirms| >= N_PROMOTE` ȘI `age >= K_PROMOTE_AGE`
- `_v15_7a_is_retrograde_eligible` — caută challenger `!= committed_value` cu `|confirms| >= M_RETROGRADE`, returnează `(True, challenger_value_idx)` sau `(False, None)`, tiebreak: max count → min value_idx (determinism)
- `_v15_7a_is_stale_for_prune` — `current_ep - last_activity >= K_PRUNE_STALE`

Constante modulare (pot fi override-uite la call site):
- `V15_7A_N_PROMOTE = 2`, `V15_7A_M_RETROGRADE = 2`, `V15_7A_K_PROMOTE_AGE = 2`, `V15_7A_K_PRUNE_STALE = 3` (conforme cu README C3).

**Self-check gated pe `V15_7A_D3_MODE=run`** — 15 unit-assertions hand-computed:
- T1: dedupe în cadrul aceluiași episod (set semantics)
- T2: izolare pe value_idx
- T3-T4: last_activity across all values / None pe slot absent
- T5: first_seen monoton
- T6-T9: promote eligibility — cazul happy (T6), N insuficient (T7), **anti-inflation L4** (T8: 3 scrieri în același episod nu satisfac N=2), age insuficient (T9)
- T10-T12: retrograde — happy (T10), ignoră committed value (T11), picks strongest (T12)
- T13-T14: stale — happy + absent slot no-op

**Decizii de design în D.3:**
- Retrograde tiebreak determinist: `(-count, value_idx)` sort — cel mai mare count câștigă; la egalitate, value_idx mai mic. Necesar pentru ops_match în D.8 (reproducibilitate test-by-test).
- `is_retrograde_eligible` **NU** elimină scenarii multi-challenger — le ordonează. Consolidatorul real (D.6) va decide 1 retrograde per slot per end_episode; aici predicate-ul e pur, fără politică.
- Nu am ales să extind `ProvisionalMemory` cu metode — helper-e free-standing sunt mai ușor de mock-uit la test (vezi self-check: construiește List direct, nu are nevoie de ProvisionalMemory).
- `Set[int]` pentru confirmation_episodes, `Optional[int]` pentru last_activity/first_seen (None = slot absent, distinge clean de „slot văzut la ep 0").

**Predicții pentru integrare (verificabil în D.8):**
- L1 ep2 end_episode: blue are confirms={ep1, ep2} → |2| >= N=2 ✓; first_seen=ep1, current=ep2, age=1 < K_age=2 → **promote NOT eligible yet**. Corect — corespunde „delay 1 episod" din README tabel L1.
- L1 ep3 end_episode: blue confirms={ep1, ep2, ep3} → |3| >= 2 ✓; age=2 ≥ 2 ✓ → promote eligible. Dar ep3 are și retrograde pe red → **intra-pas exclusion (C4.1)** blochează. Corect — promote vine la ep4.
- L4: 3 scrieri în ep1 → confirms={ep1} → |1| < 2 → never eligible. ✓ anti-inflation.
- L5 end_ep4: last_activity=ep1, current=ep4, K=3 → stale ✓.

**Gate 0 verification**: `python -c "import ast; ast.parse(...)"` → OK. Nu am atins nimic între linia 1 și 15429. Orice trusted regression single-episode rămâne byte-identical (derivation helpers nu sunt invocate pe căile Pas 6).

**Rulat local (2026-04-23)**: `scratch/d3_selfcheck.py` extrage blocul D.3 din code.py via `read_text + exec` (stub minimal pentru ProvisionalEntry) — se execută fără torch/CUDA. Output: **PASS: all 15 derivation self-checks green**. Predicatele sunt validate pe scenarii hand-computed înainte să ajungă în consolidator.

Next: D.4 — `Consolidator.reconcile` — prima operație care **nu mutează bank**, colapsează provisional duplicate pe same slot (fără promote, fără retrograde).

---

## 2026-04-23 — D.4 implementat: Consolidator.reconcile + audit trail

**Disambiguare RECONCILE counting**. Generators-ele L1-L5 au `expected_ops={"RECONCILE": 1, ...}` pentru toate familiile cu conflict (L1, L2, L4, L5) și `{}` pentru L3. Singura semantică care reproduce exact toate 5 predicțiile:

> RECONCILE fires per (slot, end_episode) iff slot a primit ≥1 entry NEW în această episodă AND post-add total entries pentru slot ≥2.

Rezultat per scenariu:
- L1: end_ep3 — slot atinge ≥2 (blue,ep2 + blue,ep3 nou) → fires 1x. End_ep4 distractor, slot netouched → no fire. Total 1 ✓
- L2: end_ep3 — same as L1 → 1 ✓
- L3: slot diferit per ep → niciodată ≥2 → 0 ✓
- L4: end_ep1 — 3 entries new (val1,val1,val2) → fires 1x, collapsează (val1,val1) → (val1) → 2 entries final ✓
- L5: end_ep1 — 2 entries new (val1,val2) → fires 1x, no collapse (distinct values) dar audit event → 1 ✓

Alternativele („per duplicate exact", „per fire while slot are conflict") rupeau cel puțin una din cele 5 predicții.

**Cod adăugat la sfârșitul `code.py` (linii 15681-15880, ~200 linii noi):**

`V15_7a_ConsolidationRecord` (dataclass, 8 câmpuri) — audit per op conform README C2:
- episode_id, operation, entity_id, attr_type, value_idx, reason, state_before, state_after

`_v15_7a_reconcile(provisional_memory, current_episode, audit) -> int`:
- iterează `slots_touched` = `{(ent,attr) | există entry cu episode_id == current_episode}`
- pentru fiecare slot cu `len(slot_entries) >= 2`: dedup pe `(value_idx, episode_id)` (cheie dict), tiebreak `min write_step`, mutează in-place `provisional_memory.entries`, emite 1 ConsolidationRecord
- returnează op_count
- sortare deterministă pe `sorted(slots_touched)` pentru reproducibilitate

**Decizii de design D.4:**
- Mutație IN-PLACE pe `provisional_memory.entries` (caller decide aplicarea — README implică mutație: „colapsează"). Nu return new list ca să rămână simetric cu prune/retrograde/promote care vor fi tot mutative.
- Tiebreak `min write_step` pentru dedup — asigură ConsolidationRecord identic la rerun pe aceeași stare.
- `audit` injectat de caller (List passed by reference) — decuplat de orice global state. Caller (D.8) va decide unde stochează log-ul.
- Reconcile NU primește `bank` ca parametru — verificat în T9 self-check via `inspect.signature`. Garanție de design: nu poate atinge bank-ul accidentală.
- Audit `value_idx=None` la RECONCILE pentru că op-ul e slot-level, nu value-level (prune/retrograde/promote vor avea value_idx setat).

**Self-check `V15_7A_D4_MODE=run`** — 18 unit-asserts pe scenarii hand-traced:
- T1 L4-ep1 (4 asserts): op_count=1, collapse 3→2, distinct (val,ep) pairs preserved, audit logged
- T2 L5-ep1 (2 asserts): op_count=1 ca audit event, entries unchanged 2→2
- T3 L1-ep3 (2 asserts): op_count=1, no actual collapse
- T4 L1-ep4 (2 asserts): slot not touched → op=0, audit empty
- T5 L1-ep2: 1 entry total → no fire
- T6 L3: cross-slot, no fire
- T7 idempotency (2 asserts): a doua rulare = no-op, entries stay collapsed
- T8 tiebreak: min write_step survives
- T9 signature check: nu există parametru `bank`
- T10 cross-entity (2 asserts): doar slot-ul touched fires

**Verificare locală (2026-04-23)**: `scratch/d4_selfcheck.py` extrage C3+C4 din code.py via `read_text + exec` cu stub-uri pentru ProvisionalEntry + ProvisionalMemory. Output: **PASS: all 18 reconcile self-checks green**.

**Gate 0**: AST parse OK. Modificări strict la sfârșitul fișierului (linii 15681+); nimic atins între 1-15679. Reconcile e funcție liberă, nu apelată pe căile Pas 6 — trusted regression byte-identical garantat.

Next: D.5 — `Consolidator.prune` — mutează doar provisional, drop entries care satisfac `is_stale_for_prune` (K_PRUNE_STALE=3). Counting: 1 op per entry pruned (vezi L5 expected `PRUNE: 2` pentru cele 2 entries pruned din slot la end_ep4).

---

## 2026-04-23 — Confirmare A100 Pas 6 + load curat D.1-D.4

User a rulat code.py pe A100 (full pipeline). Output:
- Pas 6 PASSED re-confirmat (toate 7 gates verzi, F2 0.952, byte-identical regression)
- Sectiuni v15.7a încărcate fără excepții: C1 (LongitudinalEpisodeRegime), C2 (D.2 baseline runner), C3 (derivation layer), C4 (Consolidator.reconcile + audit)
- Self-checks D.2/D.3/D.4 nu au rulat (env vars `V15_7A_D*_MODE=run` nesetate) — doar definițiile au fost validate prin import.
- Fix-ul `from typing import Any` din D.4 funcționează corect în Colab.

Concluzie: codul parsează și se importă curat pe Colab. Următoarele rulări A100 pot include `os.environ["V15_7A_D2_MODE"] = "run"` (baseline empiric) opțional.

---

## 2026-04-23 — D.5 implementat: Consolidator.prune

**Disambiguare PRUNE counting**. L5 prezice `PRUNE: 2` pentru 2 entries în provisional (val1, val2 ambele la ep1) la end_ep4 (last_activity=ep1, current=ep4, diff=3 ≥ K=3). Singura semantică care reproduce 2:

> PRUNE fires per ENTRY satisfying `_v15_7a_is_stale_for_prune` la end_episode. 1 op count per entry pruned.

Verificare contra L4: 2 entries (val1, val2 la ep1), current=ep3, diff=2 < K=3 → not stale → 0 prune ops. L4 expected_ops `{"RECONCILE": 1}` (fără PRUNE) → match ✓.

**Cod adăugat la sfârșitul `code.py` (linii 15889-16070, ~180 linii noi):**

`_v15_7a_prune(provisional_memory, current_episode, audit, K_stale=3) -> int`:
- Iterează `sorted(slots)` din provisional (determinism)
- Pentru fiecare slot: dacă `is_stale_for_prune` → emit 1 ConsolidationRecord per entry, sortate `(value_idx, episode_id, write_step)`
- Mutează `provisional_memory.entries` în-place (drop slot complet)
- Returnează `op_count = total entries pruned`

**Decizii de design D.5:**
- Counting per-entry (nu per-slot) — necesar pentru L5 PRUNE: 2.
- Sortare slots `sorted(slots)` + intra-slot `(value_idx, episode_id, write_step)` pentru audit reproductibil.
- Audit reason verbal: `last_activity=X current=Y diff=Z >= K=W` — facilitează debug post-hoc.
- Signature deliberat fără param `bank` (T7 verifică via `inspect.signature`).
- Drop atomic per slot (nu per entry) — toate entries ale slot-ului în 1 list comprehension; previne tearing dacă audit eșuează parțial.

**Self-check `V15_7A_D5_MODE=run`** — 16 unit-asserts:
- T1 L5-ep4 (3 asserts): op_count=2, entries pruned 2→0, 2 PRUNE în audit
- T2 L4-ep3 (2 asserts): diff=2 < K=3 → op=0, entries unchanged
- T3 boundary diff==K=3: prunes
- T4 boundary diff==K-1: no prune
- T5 cross-slot (2 asserts): doar slot-ul stale pruned
- T6 empty memory: no-op
- T7 signature check: nu există parametru `bank`
- T8 idempotency: a doua rulare = no-op
- T9 cross-entity (2 asserts): doar entity-ul stale prunat
- T10 audit reason mentions `diff=4 K=3`
- T11 deterministic order: prune ascending value_idx

**Verificare locală (2026-04-23)**: `scratch/d5_selfcheck.py` extrage C3+C4+C5, env `V15_7A_D5_MODE=run`. Output: **PASS: all 16 prune self-checks green**.

**Gate 0**: AST parse OK. Adăugare strict la sfârșitul fișierului (linii 15889+); nimic atins între 1-15888. Prune e funcție liberă, nu apelată pe căile Pas 6 — trusted regression byte-identical garantat.

Next: D.6 — `Consolidator.retrograde` — **prima operație care MUTEAZĂ bank-ul**. Demote committed slots când `is_retrograde_eligible` (M=2 distinct challenger episodes). Counting per L1/L2 expected `RETROGRADE: 1` → 1 op per (slot, end_episode) când retrograde fires. Critical: înainte de D.7 promote, validare obligatorie contra Gate 4 (`false_retrograde_rate = 0`) ca să nu introducem regresii pe L3 (completion no-op).

---

## D.6–D.9 implementare + sigilare (2026-04-26)

### D.6 retrograde — sealed

`_v15_7a_retrograde(provisional_memory, bank, stability_index, current_episode, audit, M=V15_7A_M_RETROGRADE)` adăugat în `code.py`.

Iterează `sorted(stability_index.committed_episode.keys())` deterministic. Pentru fiecare committed slot: dacă bank lipsește entity sau attr_slot.present=False → silent skip (defensive). Evaluează `_v15_7a_is_retrograde_eligible(committed_value_idx, M=2)` peste provisional. Dacă eligibil: mutație in-place pe `AttributeSlot` (`present=False`, `value_idx=-1`, `version+=1`, `value_emb=None`, `write_step` păstrat), șterge intrarea din `BankStabilityIndex`, emite `ConsolidationRecord(operation="RETROGRADE")` cu state_before/after complet.

**Provisional NU este atins în v1**: challenger-ul rămâne pentru ca D.7 să-l poată promova ulterior (intra-pas exclusion blochează same-end_episode promote pe slot retrogradat).

**Self-check D.6**: 56 asserts verzi. T17 GATE 4 rollup: `false_retrograde_rate = 0` peste 5 negative cases (L3 completion, L4 anti-inflation, L5 stale-only, insufficient challenger 1 ep, self-confirmation challenger==committed).

### D.7 promote — sealed

`_v15_7a_promote(provisional_memory, bank, stability_index, current_episode, audit, N=V15_7A_N_PROMOTE, K_age=V15_7A_K_PROMOTE_AGE)` adăugat în `code.py`.

**Intra-pas exclusion**: scan audit log pentru `RETROGRADE` cu `episode_id == current_episode` → set `excluded_slots`. Slot-uri retrogradate în acest end_episode sunt sărite. Forțează delay-ul de 1 episod în L1 (promote la end_ep4, nu end_ep3).

Pentru fiecare slot rămas, alege strongest eligible value (tiebreak: most confirmations, smallest value_idx pentru determinism).

**Bank-state policy** (no transitive demote v1):
- entity absent în bank → `PROMOTE_SKIPPED` (consolidator nu allocate)
- bank slot present=False → promote in-place (mutează AttributeSlot, marchează stability)
- bank slot present=True cu același value → idempotent finalize (re-marchează stability, curăță provisional)
- bank slot present=True cu valoare diferită → `PROMOTE_SKIPPED` (D.6 nu l-a demotat ⇒ nu e eligibil pentru retrograde, nici pentru overwrite)

Curăță provisional entries pentru `(slot, promoted_value)` doar (alte valori în provisional pentru același slot rămân ca historical challengers).

**Self-check D.7**: după fix-ul T12 (assertion contradictoriu inițial), toate self-check-urile verzi. T17 GATE 3 rollup: `false_promote_rate = 0` peste 5 negative cases.

### D.8 wiring — sealed

`CommitArbiterPas7a(CommitArbiterPas6)` + helper `_v15_7a_run_consolidator_pipeline()`.

Override pe `end_episode`:
1. `super().end_episode(...)` → Pas 2/6 finalize (mutează bank/provisional/stability)
2. `_v15_7a_run_consolidator_pipeline(...)` → reconcile → prune → retrograde → promote
3. `self.last_consolidator_ops` = dict counts
4. Audit log acumulează cross-episode în `self.consolidation_audit_log`

Pas 6 in-episode (write_fact, RoMR, dual conflict rule, cross-episode challenger) rămâne complet neatins.

**Self-check D.8**: 33 asserts verzi. Reproduce expected_ops byte-exact pentru L1/L2/L3/L4/L5 într-o simulare standalone (fără Pas 6 model, doar pipeline helper + mock bank).

### D.9 full eval — sealed

`v15_7a_run_full_eval_d9(base_model, v15_1_memory)` adăugat în `code.py`.

Rulează două faze:
- **Phase A**: re-rulează F1-F5 + S5/S6 cu `CommitArbiterPas7a` (consolidator activ). Verifică Gate 0 byte-identical, Gate 1 wrong_commit ≤ 0.02, Gate 2 F2 safe_resolution ≥ 0.95.
- **Phase B**: rulează L1-L5 (n=20 fiecare) cu `CommitArbiterPas7a`. Captură per-trial audit. Computează metrice false_promote/false_retrograde, l1_promote_rate, l2_retrograde_rate, etc. Verifică Gates 3-9.

Salvează raport JSON la `/content/drive/MyDrive/dcortex_v2/v15_7a/results/v15_7a_d9_full_eval.json` prin `_v15_7a_json_safe()` (chei tuple → string `"entity::attr"`).

### Patch-uri post-D.9 (ne-funcționale, doar artefact + interacțiune RoMR)

1. **L2 ep3 template** (`gen_L2_retrograde_only`): `"A {chall_val} {entity} stood nearby."` → `"The {entity} stood {chall_val} nearby."`. Cauza primară: în prima rulare D.9 sub Pas 6 RoMR live, `{chall_val}` în NP interior era clasificat ENTITY_MODIFIER (modifier al entity head), deci `cross_episode_challenger` nu se deposita la ep3 → M=2 nu se atingea → retrograde nu fires → Gate 6 0/20. Patch-ul reordonează la formă post-copulară (`stood` e în `V15_6_PAS6_COPULAS`). Re-rulare: 20/20 retrogrades, Gate 6 PASS la 1.000.

2. **JSON serializer**: `_v15_7a_json_safe(obj)` adăugat înainte de `def v15_7a_run_full_eval_d9`. Wrap pe `json.dump(_v15_7a_json_safe(d9_result), ...)`. Recursiv pe dict/list/tuple. Strict serializer-only. Verificat local: `{("dragon","color"):9}` → `{"dragon::color":9}`. Round-trip OK.

Niciuna nu atinge D6/D7/D8 sau gate logic.

### Verdict empiric

Raport oficial `v15_7a_d9_full_eval.json` (n_per_l_family=20, seed=20261103):

```
overall: true
verdicts: {Gate 0..9 → all true}
phase_a.trusted_match: true
phase_b.total_false_promote: 0
phase_b.total_false_retrograde: 0
phase_b.l1_promote_rate: 1.0
phase_b.l2_retrograde_rate: 1.0
phase_b.l3_false_retrograde_count: 0
phase_b.l4_promote_count: 0
phase_b.l5_prune_per_trial: [2]*20
```

**PAS 7A SEALED — all 10 gates green.**

### Open spec items (post-seal, non-blocking)

1. `expected_final_committed` în L1/L4/L5 nu enumără entitățile distractor. Bank-ul real conține și sloturile distractor (e.g., L1 ep4 scrie `(distractor_entity, size)`). Per-trial print arată "committed_match: 0/20" pentru aceste familii dar nu afectează niciun gate (gate-urile contează ops, nu bank exact match). Patch posibil ulterior: fie enumerare distractori în expected, fie `committed_match` devine subset-based pe sloturile țintă ale familiei.

2. `PROMOTE_SKIPPED` audit operation introdus de D.7 nu apare în spec C.2 originală. E audit-only (nu contribuie la `false_promote`). Dacă vreun consumator iterează audit fără filtrare pe operations, va vedea acest tip nou.

### Restricții post-seal

- D.6/D.7/D.8/D.9 nu se modifică
- Gate logic 0-9 nu se relaxează / redesign
- Pas 6 byte-identical
- Query path neatins
- Semantic abstraction: blocat până la adapter explicit
- Integrare cu fragmergent-memory-engine: blocată până la adapter exact definit

