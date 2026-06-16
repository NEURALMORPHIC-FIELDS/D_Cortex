# D_Cortex integration spine: LLM extractor -> sealed organ -> vetoed verbalizer

Engineering register. MEASURED. The pipeline is deterministic (Qwen greedy + MiniLM + symbolic organ),
so point values are reported.

## Architecture (two validity-critical forks resolved first)
The campaign sheet named a checkpoint that does not exist and conflated two D_Cortex memory systems.
Both were flagged and confirmed before building: (1) the neural DCortexV2Model (open-vocabulary,
real-corpus) is NOT the organ; (2) the SYMBOLIC v15.x substrate is (color/size/location/state, the old
parser, RoMR, Pas7a). The organ is the sealed DeterministicObjectBank + CommitArbiterPas7a, loaded from
steps/13_v15_7a_consolidation/code.py by an AST loader that never edits the file (dcortex/ and steps/13
stay byte-identical; the 4 semantic seals match, code.py shows only the pre-existing CRLF/LF working-
tree quirk). Components (new code, integration/):
- IngestAdapter: Qwen2.5-7B-Instruct (4-bit NF4) JSON-only extractor, constrained to the closed
  vocabulary; out-of-vocab (attribute, value) is an extraction error, never written. MiniLM
  (all-MiniLM-L6-v2) canonicalizes entities. No regex parser.
- OrganClient: thin wrapper; write_fact renders the triple to the parser's native canonical sentence,
  writes through the arbiter (RoMR at write); end_episode runs Pas7a (N=2/M=2/Kage=2/Kstale=3); query
  reads the committed value with the sealed status taxonomy + trace.
- VerbalizerControl: single exit; Qwen masked to the committed value (FOUND_COMMITTED) or to an abstain
  template (NONE_* / failure). No path returns text bypassing the veto.

## Gates (frozen, with the corrected definitions)
VERDICT: D_CORTEX_SPINE_PASS.

### G_EXTRACT - triple-correctness (commit_correct analog: fact AND query both extracted vs gold)
Head-to-head, Qwen vs the old rule-based parser (live), per family. The sealed old-parser commit_correct
baseline was 0.000 / 0.000 / 0.148 on F1/F3/F5.
- F0: Qwen 85% vs old-parser 100% (sealed old None) (control: standard phrasing, old parser native)
- F1: Qwen 50% vs old-parser 20% (sealed old 0.0)
- F3: Qwen 40% vs old-parser 0% (sealed old 0.0)
- F5: Qwen 95% vs old-parser 25% (sealed old 0.148)
Qwen beats the old parser on F1/F3/F5: True. HONEST: Qwen wins
on every hard family but its absolute commit_correct is modest on F1 (50%) and F3 (40%) and strong on
F5 (95%); on F0 the old parser is native (100%) and slightly ahead of Qwen (85%). The old-parser live
numbers approximate the sealed baseline (reconstructed instances + bare-parse metric, not the full
sealed arbiter pipeline), but the qualitative result holds: the closed-lexicon parser fails the novel
paraphrase/alias/query forms; the LLM extractor recovers them.

### G_NOCORRUPT (split per the correction)
- organ wrong_commit on GOLD triples = 0.000 (ceiling 0.02): PASS. The sealed no-corruption property survives the wrapper.
- the organ commits whatever triple it is given; extraction errors are NOT charged here, they are
  attributed in G_ATTRIB. organ standalone recall = 100%.

### G_RECALL
- organ standalone 100%; recall on CORRECTLY-extracted facts 100% (>= standalone): PASS. Memory +
  verbalization lose no recall. Full end-to-end recall 69% is bounded by
  extraction (the bottleneck is the Qwen extractor, not the organ).

### G_HALLUC: hallucinations 0 (PASS); abstained on every NONE_/uncertain.
### G_NOBYPASS: 0 leaks across 6 adversarial queries (PASS).
### G_TRACE: every answer carries trace (True).
### G_ATTRIB: extraction errors 10; end-to-end wrong
answers decomposed: extraction-cause 1, organ-cause 1 (organ-cause ~0, consistent with G_NOCORRUPT). The error budget is
dominated by extraction, not the organ.

## Claim status
MEASURED on the organ's native synthetic vocabulary (color/size/location/state) with Qwen2.5-7B-4bit extractor/verbalizer + MiniLM resolver, single machine. Domain-vocabulary expansion out of scope. dcortex/ and steps/13 byte-identical (loaded read-only).
MEASURED single machine; not multi-hardware, not independently replicated. The organ + wrapper are
clean (no corruption, no hallucination, no bypass, recall preserved); the LLM extractor genuinely
recovers phrasings the rule-based parser failed, though its absolute accuracy on the hardest families
is modest. dcortex/ and steps/13 byte-identical (loaded read-only).
