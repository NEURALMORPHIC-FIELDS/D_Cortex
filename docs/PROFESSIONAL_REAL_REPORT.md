# Real-model professional integration: RAW vs CONTROLLED hallucination

Engineering register. Claim status governs every line. MEASURED, single environment.

## Headline (the metric that matters)
On a REAL open base model (gpt2-large, fp32, hidden dim 1280) over a
real-scale pinned WIPO IPC pack, the base model's hallucination rate WITHOUT D_Cortex versus UNDER it:

- RAW (no D_Cortex):        hallucination 98.8%, answerable grounded 0.0%
- CONTROLLED (D_Cortex):    hallucination 1.2%, recall 95.0%
- RAW vs CONTROLLED delta:  +97.7% hallucination reduction

The raw model answers IPC-title questions with confident, well-formed, WRONG titles (0% grounded:
gpt2-large knows no exact IPC titles) and never abstains. Under D_Cortex it is mechanically forced to
the committed title or to abstain, so the hallucination rate collapses to near zero.

## Target model and pack
- Real base model: gpt2-large (Hugging Face, fp32, full logits + hidden states), device cuda.
  GPT-2 family chosen so hidden states stay clean (no quantization); a real published open model that
  hallucinates plausibly on obscure facts. NOT the project's substrate.
- Pack: D_Cortex_IPCAnalyst, 380 committed facts = 192 official WIPO IPC class
  codes x (title, category), pinned by GitHub commit + file SHA, HTML/version markers cleaned. Codes
  that could not be cleaned to a valid title were excluded (no fabrication).

## Gates (frozen) - all PASS
- [PASS] G_RAW
- [PASS] G_CONTROLLED
- [PASS] G_DELTA
- [PASS] G_RECALL
- [PASS] G_ABSTAIN
- [PASS] G_FORBIDDEN
- [PASS] G_VETO
- [PASS] G_NOBYPASS
- [PASS] G_BINDER
- [PASS] G_TRACE

## Mechanical enforcement under a real model
- Constrained decoding forces the committed IPC title at the factual slot via logit masking; the raw
  model's unconstrained slot emission is recorded and is consistently wrong/garbled.
- The hard verifier vetoes plausible-but-ungrounded claims (G_VETO: an invented title 'NUCLEAR PHYSICS
  AND REACTORS' for IPC A01 is rejected because it does not match committed memory) and forbidden legal
  conclusions (G_FORBIDDEN). G_NOBYPASS: zero ungrounded leaks across the adversarial battery.
- Integrity fix during the run: the verifier's contamination check first false-vetoed legitimate
  answers (an IPC title contains common words that are other facts' values); it was corrected to flag
  only a DIFFERENT entity's SAME-attribute value. A case bug that leaked a committed code into the
  'unanswerable' set was also fixed. Both were corrections, not weakenings (the wrong-fact veto is
  the claim-grounding check, untouched).

## Neural binder two-regime benchmark (honest negative)
- structured regime: binder 55.0% vs exact (entity,attribute) lookup 95.0%
  -> lookup wins decisively; the binder is near chance (50%).
- unstructured regime (paraphrased titles): binder 65.0% vs exact lookup
  0.0% (structurally inapplicable to paraphrase) vs FAIR token-overlap fuzzy
  lookup 96.7%. Against the fair fuzzy baseline the binder LOSES by
  -31.7pp.
- Integrity note: an initial run judged the binder against the 0% exact lookup and showed a misleading
  +65pp binder 'win'. That was a strawman (exact lookup cannot match paraphrases); a fair fuzzy lookup
  baseline was added, against which the binder loses.
- WHY: the content-addressed binder can only ground what the base model's representations already
  encode. gpt2-large does not encode obscure IPC code<->title relations, so the binder is near chance
  exactly where grounding matters most. The binder validated at 99.3% on the D_Cortex substrate
  (geography it was warm-started on) does NOT transfer to gpt2-large IPC hidden states.

## Claim separation (what grounded what)
- The RAW->CONTROLLED hallucination collapse is delivered ENTIRELY by deterministic (entity,attribute)
  lookup + constrained decoding + the hard verifier veto. The neural binder contributes nothing here
  and is reported as a measured negative. No grounding is attributed to the binder.

## Claim status
VERDICT: D_CORTEX_REAL_PROFESSIONAL_PASS. MEASURED on a real open model (gpt2-large, fp32) and a real-scale pinned
WIPO IPC pack, single environment (CUDA, same machine, single organization). Mechanical grounding
reduces hallucination from 99% (raw) to 1.2%
(controlled). SCOPE / NOT PROVEN: one real model, one pack, one machine; multi-hardware reproduction
and independent replication remain for any production or responsibility-critical claim. A low
controlled hallucination rate here is a strong MEASURED result, not a deployment guarantee.

## External citation and usage constraint (binding)
This result MUST NOT be cited externally (patent / EESR / regulatory / marketing / public claims) until
BOTH of the following hold: (1) the same benchmark has been run on a PRODUCTION-grade base model (not
gpt2-large), and (2) the caveats in this report are carried with the citation. Until then this stays on
the feature branch and is NOT to be merged into main.

Caveats that must travel with any citation:
- MEASURED, single environment: one real model (gpt2-large, fp32), one pack (D_Cortex_IPCAnalyst, 192
  WIPO IPC codes), one machine (CUDA), one organization. NOT multi-hardware, NOT independently replicated.
- The grounding is delivered by deterministic (entity,attribute) lookup + constrained decoding + the hard
  verifier veto. The neural binder is a MEASURED NEGATIVE here (near chance on gpt2-large IPC hidden
  states; loses to a fair fuzzy lookup). No grounding is attributed to the binder.
- gpt2-large is a small model; its 98.8% raw hallucination on obscure IPC facts may differ from a
  production model's rate. The RAW vs CONTROLLED delta must be re-measured on the production model.
- Bug provenance (traceability): the verifier contamination false-veto fixed here was PRE-EXISTING and
  latent in the prior pushed branch feature/professional-integration @7e45902; that prior campaign was
  UNAFFECTED (its own committed_recall = 1.0, G3 PASS) and its verdict is unchanged. The case bug was
  INTRODUCED and fixed within this campaign's new harness. See .claude/project_log.json.
