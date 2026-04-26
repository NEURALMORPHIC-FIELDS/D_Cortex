# Step 08-10 — v15.5 External Holdout → v15.6 Pas 3 EntitySpanComposer

**Status**: SIGILAT (istoric, bundle concatenat).

## Ce conține

`code.py` (35.870 linii) este concatenarea exportată din Colab a trei celule de lucru succesive:

| Logical step | Versiune | Scop |
|---|---|---|
| step 8  | v15.5                | External holdout generator (F1–F5, S5, S6) + evaluator robustness |
| step 9  | v15.6 Pas 1          | InternalizationPacket + structural wrapper + equivalence test |
| step 10 | v15.6 Pas 3          | EntitySpanComposer rule-based + F2 breakdown scorer |
| step 10.1 | v15.6 Pas 3.1a     | F2 causal diagnosis offline (verdict: PAS 3.1 FALSIFIED) |

Toate cele 4 bucăți rulează secvențial în același fișier. Nu se poate separa fără a sparge artefactul istoric — de aceea rămâne ca bundle.

## Ce era sub capotă (stratificat)

- **v15.1 substrate**: `DeterministicObjectBank`, entity_id canonicalization, parse_fact/query rule-based
- **v15.1 shadow**: `ShadowAttributeRouter`, `ShadowTypedValueHeads`, `ShadowObjectResolver` (frozen oracle pentru pașii următori)
- **v15.2 protocol**: `ParsePacket`, `ParseVerifier`, 7 `AmbiguityFlag`, `PARSE_UNCERTAIN` ca al 5-lea read status
- **v15.4 extinderi**: trigger families, query patterns, `ATTR_WEAK_SIGNAL`, `PREFIX_ALIAS_MAP` pentru multi-BPE-token entities
- **v15.5 holdout**: 5 familii (F1 paraphrase syntax, F2 multiword, F3 novel alias, F4 discourse intercalation, F5 novel query forms) + S5/S6 ambiguity probes
- **v15.6 Pas 1**: `InternalizationPacket` (13 câmpuri Slide 9), `CommitPath` enum, wrapper echivalent byte-identic cu v15.4.1
- **v15.6 Pas 2**: `ProvisionalMemory`, `EpisodeBuffer`, `BankStabilityIndex`, `CommitArbiter`, `ReadArbiter`, `FOUND_DISPUTED`
- **v15.6 Pas 3**: `EntitySpanComposer` (9 premodifiers whitelist, max 2 modifiers, overlap → UNCOMPOSED), `CommitArbiterPas3`
- **v15.6 Pas 3.1a**: 8 causal labels per F2 trial, 3 gain metrics counterfactual, verdict: `gain_symmetric / failures < 50%` → Pas 3.1 FALSIFIED → redirect spre Pas 6

## Mediu de execuție

Google Colab A100 (bfloat16, TF32). Workspace pe Drive: `/content/drive/MyDrive/dcortex_v2/`.
Shadow checkpoint: `v15_2/checkpoints/shadow_final.pt`.

## De ce NU se modifică acest folder

- Este oracle pentru Pas 6 (echivalența Pas 1 verifică că wrapper-ul nu deviază de la v15.4.1)
- Este referință empirică pentru deltele raportate în pașii următori
- Conține diagnosticul care a justificat arhitectural trecerea la Pas 6 (nu Pas 3.1)

Dacă un bug este descoperit aici, corecția apare în `steps/NN_.../` următor, nu aici.

## Verdict istoric

- **v15.5 external holdout**: benchmark closure only (5/5 families ≥ 85% NU a fost atins)
- **v15.6 Pas 1**: PAS 1 EQUIVALENCE PASS (500 trials, 0 divergences bank+read)
- **v15.6 Pas 3**: PAS 3 PASSED — F2 primar ≥ 95% safe_resolution, wrong_commit = 0
- **v15.6 Pas 3.1a**: diagnostic FALSIFICAT (gain_symmetric < 50% failures) → redirect către Pas 6
