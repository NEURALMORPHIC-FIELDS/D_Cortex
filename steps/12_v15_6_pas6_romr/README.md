# Step 12 — v15.6 Pas 6 — Role-of-Modifier Resolver (RoMR)

**Status**: SIGILAT. Verdict: **PAS 6 PASSED pe A100 (toate 7 gates)**.

## Problema atacată

F2 `multiword_entities` rămânea la 21.8% uncertain după Pas 3 (EntitySpanComposer). Diagnosticul Pas 3.1a a arătat că reziduul NU era o problemă de internalizare entity-side (simetrie write/read), ci o problemă de **rol semantic al valorilor** — aceeași familie atributivă („small", „huge") putea apărea atât ca modifier de entitate, cât și ca valoare predicativă.

Token-ul „small" în `The small horse` NU e valoare de size; e modifier de entitate.
Token-ul „small" în `The horse is small` E valoare de size.

Lexical, identici. Structural, diferiți. Parser-ul v15.4 nu avea această distincție.

## Schimbări față de Pas 3

### Componente noi

1. **`RoleOfModifierResolver`** — clasifică fiecare `value_candidate` ca:
   - `ENTITY_MODIFIER` (în NP span, înainte de head)
   - `ATTRIBUTE_VALUE` (post-copula sau attributive între head și copula)
   - `UNCERTAIN` (în afara NP span pre-head sau fără copula)

2. **Packet-level `REAL_CONFLICT`** — relație între două ATTRIBUTE_VALUE candidate din aceeași familie (nu proprietate a unui singur token):
   - Detectat **înainte de filtrare**, pe raw candidates per family
   - Dacă ≥ 2 valori distincte în aceeași familie → promovează TOȚI candidații la ATTRIBUTE_VALUE
   - Verifier-ul vede ambele → `ATTR_CONFLICT_STRONG` → `PARSE_UNCERTAIN` onest

3. **Recompute flag după filtrare**:
   - **Value-dependent flags** (invalidate și re-derivate): `MULTIPLE_ATTR_TRIGGERS`, `ATTR_CONFLICT_STRONG`, `ATTR_VALUE_MISMATCH`, `VALUE_MISSING_OR_UNCLEAR`
   - **Independente** (preserved): `TEMPLATE_UNKNOWN`, `REFERENT_AMBIGUOUS`, `ENTITY_SPAN_AMBIGUOUS`, `OP_TYPE_AMBIGUOUS`, `MULTI_ENTITY_SAME_TYPE`, `ATTR_WEAK_SIGNAL`, `MULTI_FAMILY_COMPETITION`

4. **Filtrare coerentă a `attribute_candidates`** cu filtered `value_candidates` — rezolvă un bug descoperit local unde v15.4 infera attribute candidates din value tokens, iar după filtrarea value-ului spurios atributul rămânea fără suport → `_top_attribute` alegea o familie fără valoare → `PARSER_FAILURE`.

5. **NP span independent de Pas 3 composer**:
   - Pas 3 composer e conservator (doar 9 premodifiers whitelist) — corect pentru entity composition, dar insuficient pentru role labeling
   - RoMR construiește propriul NP span: walk backward de la head până la determiner / blocker / copula / alt head. Orice alt cuvânt intră în NP.

6. **Integrare**: `CommitArbiterPas6` extinde `CommitArbiterPas3`:
   - RoMR rulează DUPĂ `v15_4_parse_fact`, ÎNAINTE de `V15_4_VERIFIER.verify`
   - Shallow copy al packet-ului; raw packet preservat pentru audit
   - Filtered packet trece prin verifier
   - Audit trail complet în `filtered_pkt.parser_evidence["romr"]`: raw vs filtered candidates, raw vs recomputed flags, packet_level_conflict, entity_span_used, head_word_pos, copula_word_pos, contoare per label

### Invarianți menținuți

- Substrate v15.1: neatins
- Parser v15.4 general: neatins
- Shadow checkpoint: neatins (frozen oracle din `v15_2/checkpoints/shadow_final.pt`)
- Query path: neatins (RoMR e fact-only)
- Pas 2 arbiters (ProvisionalMemory, EpisodeBuffer, BankStabilityIndex): neatinși
- Pas 3 EntitySpanComposer: neatins (RoMR rulează în paralel pentru role, composer decide head-ul)
- `wrong_commit`: rămâne 0 pe toate familiile

## Rezultate (A100, n=500 per family, n=200 per S-probe)

### F2 multiword_entities (target primar)

| Metric | Pas 3 baseline | Pas 6 A100 | Delta |
|---|---:|---:|---:|
| commit_correct | 0.782 | **0.952** | +0.170 |
| provisional_correct | 0.000 | 0.000 | — |
| uncertain | 0.218 | 0.048 | −0.170 |
| wrong_commit | 0.000 | 0.000 | 0.000 |
| parser_failure | 0.000 | 0.000 | — |

### F2 re-diagnosis post-RoMR

- post-RoMR `attr_write_fail` count: 0 / 500 (rate = 0.000)
- trials cu `REAL_CONFLICT` detectat: 24 / 500
- total `ENTITY_MODIFIER` tokens dropped din value_candidates: 85

### Regression guards

- Trusted snapshot before/after: **byte-identical** (Gate 0 PASS)
- S5 `conflict_intercalated`: honesty = 1.000, overcommit = 0.000
- S6 `entity_competition_cross`: honesty = 1.000, overcommit = 0.000
- F4 `discourse_intercalation`: commit_correct = 1.000
- F1/F3/F5: IDENTIC cu Pas 3 (scope guard confirmat, fără contaminare)

### Toate 7 gates

✓ Gate 0: trusted regression byte-identical
✓ Gate 1: wrong_commit ≤ 2% pe toate 5 familii (toate 0.000)
✓ Gate 2: F2 safe_resolution ≥ 0.95 (0.952)
✓ Gate 3: F2 wrong_commit == 0 (strict)
✓ Gate 4: S5/S6 honesty ≥ 0.95, overcommit ≤ 0.02
✓ Gate 5: F4 safe_resolution ≥ 0.99 (1.000)
✓ Gate 6: F2 attr_write_fail_rate ≤ 0.05 (0.000)

## Mediu de execuție

Google Colab A100-SXM4-80GB (bfloat16, TF32, SDPA). Workspace Drive: `/content/drive/MyDrive/dcortex_v2/v15_6/`.

## Artefacte

- Checkpoint rezultat: `/content/drive/MyDrive/dcortex_v2/v15_6/results/v15_6_pas6_romr.json`
- Shadow oracle (nemodificat): `/content/drive/MyDrive/dcortex_v2/v15_2/checkpoints/shadow_final.pt`

## Ce înseamnă acest pas pentru misiune

Pentru prima dată, sistemul nu doar extrage token-i, ci **decide funcția cognitivă a fiecărui token înainte de commit**. Același cuvânt primește rol diferit în funcție de poziție structurală. E prima componentă de internalizare reală — pre-memorizare, nu post-matching.

Citat relevant din MISIUNEA.txt: *nu arunci o carte pe jos și spui că „a intrat în sistem", cartea trebuie clasificată, etichetată, pusă pe raftul corect*. RoMR e exact asta pentru valorile atributive.

## Succesor prevăzut

`13_...` — un strat de consolidare / replay / promotion-retrograde. Momentul în care memoria începe să opereze asupra ei înseși. Nu un semantic resolver pentru F1/F3/F5 — acela vine după consolidare, altfel ar fi doar un parser mai sofisticat care produce ipoteze pe care arhitectura nu le metabolizează.

Vezi `PROGRESS.md` secțiunea `Next`.
