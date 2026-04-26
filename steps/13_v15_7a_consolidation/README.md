# Step 13 — v15.7a — Consolidator sincron la end_episode

**Status**: SEALED (2026-04-26). Verdict: **PAS 7A SEALED — all 10 gates green**

D_Cortex v15.7a este primul organ longitudinal validat: memoria poate reconcilia, prune, retrograde și promote la `end_episode` fără să contamineze Pas 6.

## Verdict final D.9

Artefact oficial: `/content/drive/MyDrive/dcortex_v2/v15_7a/results/v15_7a_d9_full_eval.json`

| Gate | Threshold | Rezultat | Status |
|---|---|---|---|
| 0 | trusted regression byte-identical | snapshot before == after | PASS |
| 1 | wrong_commit ≤ 0.02 toate F1-F5 | 0.000 toate | PASS |
| 2 | F2 safe_resolution ≥ 0.95 | 0.952 | PASS |
| 3 | false_promote_rate = 0 | 0 / 100 sequences | PASS |
| 4 | false_retrograde_rate = 0 | 0 / 100 sequences | PASS |
| 5 | L1 promote_rate ≥ 0.95 | 1.000 (20/20) | PASS |
| 6 | L2 retrograde_rate ≥ 0.90 | 1.000 (20/20) | PASS |
| 7 | L3 false_retrograde = 0 | 0 / 20 | PASS |
| 8 | L4 promote_count = 0 | 0 / 20 | PASS |
| 9 | L5 prune_count ≥ 1/trial | 2 per trial × 20 | PASS |

n_per_l_family = 20, seed = 20261103. Total: 100 secvențe longitudinale × ~3-4 episoade/secvență.

## Status componente

- C1 LongitudinalEpisodeRegime — sealed
- C2 baseline runner (D.2) — sealed
- C3 derivation layer (D.3) — sealed
- C4 Consolidator.reconcile (D.4) — sealed
- C5 Consolidator.prune (D.5) — sealed
- **D6 Consolidator.retrograde — sealed**
- **D7 Consolidator.promote — sealed**
- **D8 CommitArbiterPas7a wiring — sealed**
- **D9 full evaluator — sealed**

Toate componentele frozen. Nu se atinge nimic în Pas 7a fără un Pas 7b/8 explicit deschis.

## Patch-uri minore aplicate post-D9 (ne-funcționale pentru critical path)

1. **L2 ep3 template** — schimbat de la `"A {chall_val} {entity} stood nearby."` (NP interior, RoMR-filtrat ca ENTITY_MODIFIER) la `"The {entity} stood {chall_val} nearby."` (post-copular, ATTRIBUTE_VALUE). Rezolvă Gate 6 sub Pas 6 RoMR live. Cod: `gen_L2_retrograde_only` în `code.py`.
2. **JSON serializer fix** — helper `_v15_7a_json_safe()` convertește chei tuple `(entity_id, attr_type)` în string `"entity::attr"` la serializarea raportului D9. Strict serializer-only, nu schimbă semantica evaluării.

Niciuna nu atinge D6/D7/D8 sau gate logic.

## Restricții post-seal

- **Pas 6**: byte-identical, neatins
- **D6/D7/D8**: nu se modifică
- **Gate logic 0-9**: nu se relaxează, nu se redesign-ează
- **Query path**: neatins
- **Semantic abstraction**: NU se începe încă (decizie ulterioară: cere adapter explicit)
- **Integrare cu fragmergent-memory-engine**: NU începe (cere adapter exact definit)

## Ce este live acum

1. Memoria operează asupra propriei istorii la `end_episode`.
2. **Stabilul poate cădea**: retrograde 20/20 pe L2 confirmă demote sub evidență contrară M=2.
3. **Provizoriul poate urca**: promote 20/20 pe L1 confirmă elevation prin convergență N=2 + age=K_age.
4. **Disciplina onestității păstrată**: zero false_promote, zero false_retrograde, F2/F4/S5/S6 neatinse.
5. **Pas 6 critical path byte-identical**: trusted regression intact.

## Problema atacată

Arhitectura actuală (până la Pas 6 inclusiv) operează memoria la nivel de frontieră de episod: write, read, commit, provisional. Nu există **niciun mecanism prin care memoria operează asupra propriei istorii**. Provisional entries se acumulează fără a fi vreodată promovate la committed. Committed nu e niciodată demotat chiar când evidența contrară crește peste prag.

Consecința: sistemul are zonă provizorie ca depozit pasiv, nu ca spațiu care devine stabil prin convergență sau instabil prin contestare.

## Ipoteza arhitecturală

Dacă consolidarea rulează **sincron la end_episode**, după finalize-ul Pas 2 existent, atunci:
- cauzalitatea rămâne auditabilă (fiecare schimbare a memoriei = exact o operație trasabilă la un motiv)
- infrastructura episodică existentă rămâne validă fără modificări
- sistemul dezvoltă dinamică longitudinală (promote/retrograde între episoade) fără a compromite `wrong_commit = 0`

Consolidator continuu / background = **țintă ontologică finală, NU acest pas**. Pas 7a e prima formă validabilă a memoriei care se gândește pe sine.

## Componente noi

### C1. LongitudinalEpisodeRegime

Metricile actuale (F1–F5, S5/S6) rulează cu `bank.reset()` între trials. Consolidarea nu există pe un singur episod — e operație între episoade. **Primul lucru construit, înainte de consolidator.**

Cinci familii L:

| Familie | Scenariu | Comportament așteptat |
|---|---|---|
| **L1_promote_cycle** | Committed red; 2 episoade contestare cu blue; un episod distractor | end ep3: retrograde red. end ep4: promote blue (blocat de retrograde în ep3 → delay 1 episod) |
| **L2_retrograde_only** | Committed red; 2 episoade contestare cu blue | end ep3: retrograde red. Nu ajungem la promote |
| **L3_completion** | Același entity primește color apoi size apoi location în episoade distincte | Toate 3 committed. Consolidator no-op (sloturi diferite, fără conflict) |
| **L4_no_inflation** | Conflict intra-ep1 cu red repetat (red, red, blue). Urmează 2 episoade distractor | Reconcile colapsează la {red:{ep1}, blue:{ep1}}. NICI red NICI blue nu sunt promovate (1 episod distinct, N=2 cerut) |
| **L5_stale_prune** | Conflict ep1 crează provisional. K=3 episoade fără activitate pe acest slot | end ep4: prune stale pe ambele intrări provisional |

Fiecare sequence e un `List[HoldoutEpisode]` consumat secvențial, cu bank + provisional_memory persistenți între episoade (reset doar între sequences distincte).

### C2. ConsolidationTrace — audit obligatoriu

Fiecare operație (promote, retrograde, reconcile, prune) emite un `ConsolidationRecord` cu episode_id, operation, target slot, reason, state_before/after. Fără audit trail, consolidarea devine exact cutia neagră pe care o refuzăm. Cu audit trail, fiecare schimbare e falsificabilă post-hoc.

### C3. Consolidator — 4 operații atomice

Rulează ÎN `end_episode`, DUPĂ finalize-ul Pas 2 existent. Ordine strictă:

```
end_episode():
    [existing Pas 2 finalize — unchanged]
    consolidator.reconcile(...)   # colapsează duplicate provisional pe același slot
    consolidator.prune(...)       # drop provisional stale
    consolidator.retrograde(...)  # demote committed cu evidență contra suficientă
    consolidator.promote(...)     # elevate provisional la committed
    # fără side-effect în afara bank + provisional_memory
```

**Reguli (parametri aprobați)**:

- **N_promote = 2** confirmări (distinct episode_ids)
- **M_retrograde = 2** challengers (distinct episode_ids)
- **K_promote_age = 2** episoade minim între prima apariție și promote
- **K_prune_stale = 3** episoade fără activitate → prune

**„Independent confirmation" = `episode_id` distinct** (decizie aprobată). Source_text identic între episoade rămâne audit-only în v1, nu blocant. Multiple writes în același episod = 1 confirmare, tratate via reconcile înainte de promote.

### C4. Reguli adiționale (aprobate)

1. **Intra-pas exclusion**: `promote` NU are voie să ruleze pe un slot atins de `retrograde` în același `end_episode`. Previne paradox local fără a introduce memorie temporală.

2. **No-op guarantee**: dacă există < 2 `episode_id` distincte în ProvisionalMemory pentru un slot, consolidatorul este no-op pe acel slot (explicit în cod și audit).

3. **One-slot locality**: fiecare operație e permisă DOAR pe același `(entity_id, attr_type)`. Fără raționament trans-slot, fără propagare între atribute, fără cross-entity bleed.

4. **Fără cooldown preventiv**: nu adăugăm memorie de ordin superior (cooldown post-retrograde) în v1. Dacă apare oscilație retrograde→promote→retrograde în test, atunci cooldown devine patch justificat empiric.

## Gates de acceptare (toate 10 hard)

### Invarianți de regresie (Pas 6 rămâne valid)

- **Gate 0**: trusted regression byte-identical pe single-episode tests (F1–F5, S5, S6, F4)
- **Gate 1**: `wrong_commit_rate ≤ 0.02` pe toate familiile clasice
- **Gate 2**: F2 safe_resolution ≥ 0.95 (Pas 6 nu regresează)

### Invarianți noi (consolidator onest)

- **Gate 3**: `false_promote_rate = 0` (STRICT — analogul wrong_commit pentru promovare)
- **Gate 4**: `false_retrograde_rate = 0` (STRICT — committed corect nu trebuie demotat)

### Comportament longitudinal

- **Gate 5**: L1 → `promote_rate ≥ 0.95` după 3 episoade de confirmare
- **Gate 6**: L2 → `retrograde_rate ≥ 0.90` după 3 episoade de contestare
- **Gate 7**: L3 → `false_retrograde_rate = 0` pe completări (nu sunt conflicte, sunt adăugiri)
- **Gate 8**: L4 → `promote_count = 0` când N confirmări vin din același episod (anti-inflation)
- **Gate 9**: L5 → `prune_count ≥ 1` pentru provisional stale după K=3 episoade tăcute

## Scope guard (ce NU atingem)

- Substrate v15.1
- Parser v15.4
- Shadow oracle
- RoMR (Pas 6) — neatins
- CommitArbiter (Pas 2 logic in-episode) — neatins
- ReadArbiter — neatins
- Pas 3 EntitySpanComposer — neatins
- Testele single-episode (F1-F5, S5, S6) — neatinse (Gate 0 verifică byte-identical)

Consolidatorul NU:
- modifică nimic în timpul unui episod activ
- schimbă decizia de write/read în afara end_episode
- atinge query path
- face cross-entity reasoning
- face semantic abstraction

## Ordinea de implementare (incremental testabil)

- **D.1**: LongitudinalEpisodeRegime + generatori L1–L5 (fără consolidator)
- **D.2**: Smoke test baseline: rulează L1–L5 cu Pas 6 (fără consolidator). Baseline pre-consolidator.
- **D.3**: Extend ProvisionalEntry cu `confirmation_episodes` + `last_activity_episode`
- **D.4**: `Consolidator.reconcile` (cel mai sigur — nu mutează bank)
- **D.5**: `Consolidator.prune` (mutează doar provisional)
- **D.6**: `Consolidator.retrograde` (mutează bank — validat contra Gate 4 ÎNAINTE de promote)
- **D.7**: `Consolidator.promote` (cel mai riscant — validat contra Gate 3)
- **D.8**: Wire în `CommitArbiterPas7a.end_episode`
- **D.9**: Full eval: single-episode (Gate 0–2) + longitudinal (Gate 3–9)

Între D.6 și D.7, dacă Gate 4 eșuează, **STOP și re-proiectăm regulile retrograde** înainte de a adăuga promote. Ordinea garantează că orice eșec e localizat într-o operație specifică.

## Mediu de execuție

Identic cu Pas 6: Google Colab A100-SXM4-80GB, bfloat16, TF32, SDPA. Workspace Drive: `/content/drive/MyDrive/dcortex_v2/v15_7a/`.

## Artefacte planificate

- `code.py` — cod complet (extins din Pas 6)
- `README.md` — acest document
- `NOTES.md` — jurnal intern în curs
- Results JSON: `/content/drive/MyDrive/dcortex_v2/v15_7a/results/v15_7a_consolidation.json`

## Risc principal + plan B

- **Risc #1**: `false_promote_rate > 0`. → cresc N=2 la N=3, validez promote rămâne viabil pe L1.
- **Risc #2**: cascada retrograde→promote→retrograde. → introduce cooldown empiric justificat (nu preventiv).
- **Risc #3**: consolidatorul rupe Gate 0. → consolidatorul e no-op când < 2 episoade distincte în ProvisionalMemory pentru slot (deja regulă explicită).

## Succesor prevăzut

După sigilare Pas 7a: discuție deschisă pentru:
- **Pas 7b**: transformare consolidator în proces periodic/background
- **Pas 8**: strat semantic conservator pentru F1/F3/F5, construit peste consolidator (produce ipoteze pe care memoria le metabolizează)
