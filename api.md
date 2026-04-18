# D_Cortex v2.0-alpha: API Reference

## Top-Level Classes

### DCortexV2Model

**Location**: `dcortex.model`

Main model class implementing the dual-agent architecture.

```python
class DCortexV2Model(nn.Module):
    def __init__(self, config: DCortexConfig) -> None
```

#### encode

```python
def encode(
    self,
    input_ids: torch.Tensor,                    # [B, T]
    answer_token_id: Optional[torch.Tensor] = None,  # [B]
    lexical_alpha: float = 0.9,
    force_bank: Optional[str] = None,
) -> Dict[str, torch.Tensor]
```

Run encoder (writer agent) and write to memory banks.

**Parameters:**
- `input_ids`: Fact tokens, shape [B, T]
- `answer_token_id`: Token ID of answer for lexical binding. If None, value is context-only.
- `lexical_alpha`: Weight on lexical component (0-1). 0.9 recommended for validation.
- `force_bank`: Name of bank to force write to ('state', 'episode_obj', 'conflict', 'archive', 'working'). If None, writer's gate decides.

**Returns:** Dict with keys:
- `gate_probs`: [B, 6] softmax over bank choices
- `w_value`: [B, D] written value (with gradient)
- `w_k_ent`: [B, d_ent] entity key (with gradient)
- `w_k_rel`: [B, d_rel] relation key
- `w_k_typ`: [B, d_typ] type key
- `q_ent`, `q_rel`, `q_typ`: equivalent queries (for diagnostic)
- `h_pool`: [B, D] encoder pooled output
- `addr_code`: [B, D] shared address code
- `slot_writes`: List of (bank_name, slot_idx) per batch element

#### decode

```python
def decode(
    self,
    input_ids: torch.Tensor,                    # [B, T]
    attention_mask: Optional[torch.Tensor] = None,
    force_attend: bool = False,
    return_retrieved: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
```

Run decoder (reader agent), produce logits with memory-conditioned attention.

**Parameters:**
- `input_ids`: Question tokens, shape [B, T]
- `attention_mask`: Padding mask [B, T]
- `force_attend`: If True, fusion blocks bypass mem_gate (full memory contribution)
- `return_retrieved`: If True, also returns retrieved_value for auxiliary emission

**Returns:**
- If `return_retrieved=False`: logits [B, T, V]
- If `return_retrieved=True`: (logits, retrieved_value) where retrieved_value is [B, D]

#### reset_memory

```python
def reset_memory(self) -> None
```

Clear all memory banks and reset counters.

#### begin_episode

```python
def begin_episode(self) -> None
```

Reset episode-scoped state (currently EpisodeSSM in encoder).

#### clear_overlays

```python
def clear_overlays(self) -> None
```

Clear differentiable overlays from all banks. Call after backward pass to prevent stale gradients.

#### memory_snapshot

```python
def memory_snapshot(self) -> Dict[str, Dict]
```

Return diagnostic snapshot: `{bank_name: {occupied: N, usage_sum: float, last_write: step}}`.

---

### AuxAnswerHead

**Location**: `dcortex.aux_modules`

Direct projection from retrieved value to vocabulary logits.

```python
class AuxAnswerHead(nn.Module):
    def __init__(self, config: DCortexConfig, shared_token_emb: nn.Embedding) -> None
    
    def forward(self, retrieved_value: torch.Tensor) -> torch.Tensor
    # retrieved_value: [B, D]
    # returns: [B, V] logits
```

Used for primary emission in validated configuration. Tied to `shared_token_emb`.

---

### ValueToKeyProjector

**Location**: `dcortex.aux_modules`

Projector from value space to key space for L_cycle loss.

```python
class ValueToKeyProjector(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor
    # value: [B, D]
    # returns: [B, d_ent]
```

Not used in v11 training loss (simplified to L_emit + L_sel + L_sep_neg only).

---

### SharedAddressEncoder

**Location**: `dcortex.shared_address`

Shared address encoder (C_sigma) used by both writer and reader.

```python
class SharedAddressEncoder(nn.Module):
    def forward(
        self,
        embeddings: torch.Tensor,           # [B, T, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor
    # returns: [B, D] address code
```

---

## Memory Subsystem

### MemoryBank

**Location**: `dcortex.memory.banks`

Storage for key-value pairs with overlay mechanism.

```python
class MemoryBank(nn.Module):
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        d_ent: int,
        d_rel: int,
        d_typ: int,
    ) -> None
    
    def free_slot(self) -> int
    def lru_slot(self) -> int
    def n_occupied(self) -> int
    
    def set_overlay(
        self,
        slot: int,
        value: torch.Tensor,        # with gradient
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> None
    
    def clear_overlay(self) -> None
    
    def get_diff_values(self) -> torch.Tensor   # overlay-aware [C, D]
    def get_diff_k_ent(self) -> torch.Tensor    # overlay-aware [C, d_ent]
    def get_diff_k_rel(self) -> torch.Tensor
    def get_diff_k_typ(self) -> torch.Tensor
    
    def snapshot(self) -> Dict
```

#### Persistent buffers (no gradient):
- `k_ent`, `k_rel`, `k_typ`, `values`: storage tensors
- `occupied`: bool mask
- `usage`: LRU counters
- `last_write_step`: timestamps

#### Overlay (with gradient):
Per-slot differentiable values set by writer. Readers combine overlay with buffers via `get_diff_*()` methods.

---

### QueryEngine

**Location**: `dcortex.memory.query`

Projection from address code to three key/query spaces.

```python
class QueryEngine(nn.Module):
    def __init__(self, config: DCortexConfig) -> None
    
    def forward(
        self,
        addr_code: torch.Tensor,    # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    # returns: (k_ent [B, d_ent], k_rel [B, d_rel], k_typ [B, d_typ])
```

---

### MemoryWriter

**Location**: `dcortex.memory.writer`

Write facts to memory banks with lexical binding.

```python
class MemoryWriter(nn.Module):
    BANK_ORDER = ["state", "episode_obj", "conflict", "archive", "working"]
    
    def __init__(self, config: DCortexConfig, shared_query_engine: nn.Module) -> None
    
    def forward(
        self,
        h_pool: torch.Tensor,                              # [B, D]
        addr_code: torch.Tensor,                           # [B, D]
        updater: MemoryUpdater,
        banks: Dict[str, MemoryBank],
        step: int,
        force_write: bool = False,
        answer_emb: Optional[torch.Tensor] = None,         # [B, D]
        lexical_alpha: float = 0.9,
        force_bank: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]
```

Value computation:
```
value_ctx = value_head(LayerNorm(h_pool))
if answer_emb is not None:
    value_lex = lexical_W_v(answer_emb)
    value = lexical_alpha * value_lex + (1 - lexical_alpha) * value_ctx
else:
    value = value_ctx
```

Routing:
- If `force_bank` given: force that bank
- Otherwise: argmax over `gate[:5]` (excluding skip)

---

### MemoryUpdater

**Location**: `dcortex.memory.updater`

Decide slot allocation vs update vs eviction.

```python
class MemoryUpdater(nn.Module):
    def __init__(self, config: DCortexConfig) -> None
    
    @torch.no_grad()
    def update(
        self,
        bank: MemoryBank,
        value: torch.Tensor,    # detached
        k_ent: torch.Tensor,    # detached
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
        step: int,
        is_conflict: bool = False,
    ) -> int   # slot_idx
    
    @torch.no_grad()
    def detect_conflict(
        self,
        bank: MemoryBank,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> bool
```

Logic:
1. Compute similarity of incoming keys against occupied slots
2. If free slot AND best_sim < theta_match: allocate new
3. If best_sim >= theta_match: EMA update existing slot
4. Else (no free, no match): evict LRU

---

### SemanticReader

**Location**: `dcortex.memory.readers`

Read from a single bank with temperature-scaled softmax.

```python
class SemanticReader(nn.Module):
    def __init__(self, config: DCortexConfig) -> None
    
    def forward(
        self,
        q_ent: torch.Tensor,    # [B, d_ent]
        q_rel: torch.Tensor,
        q_typ: torch.Tensor,
        bank: MemoryBank,
    ) -> torch.Tensor           # [B, D]
```

Temperature = 20 hardcoded in v10+.

---

### MemoryReadFusion

**Location**: `dcortex.memory.readers`

Combine 5 reader outputs into memory_tokens for decoder cross-attention.

```python
class MemoryReadFusion(nn.Module):
    def forward(
        self,
        r_state: torch.Tensor,      # [B, D]
        r_episode: torch.Tensor,
        r_conflict: torch.Tensor,
        r_archive: torch.Tensor,
        r_working: torch.Tensor,
    ) -> torch.Tensor               # [B, 5, D]
```

Output used by decoder fusion blocks. Note: `retrieved_value` in `decode()` is computed as raw sum BEFORE this module to avoid bias pollution.

---

## Configuration

### DCortexConfig

**Location**: `dcortex.config`

```python
@dataclass
class DCortexConfig:
    # Backbone
    hidden_dim: int = 768
    n_enc_heads: int = 12
    n_dec_heads: int = 12
    n_enc_layers: int = 4
    n_dec_layers: int = 12
    n_fusion_blocks: int = 4
    enc_ff_dim: int = 3072
    dec_ff_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 1024
    vocab_size: int = 50257
    
    # Memory banks
    state_capacity: int = 64
    episode_obj_capacity: int = 128
    conflict_capacity: int = 32
    archive_capacity: int = 512
    working_capacity: int = 16
    
    # Keys
    d_ent: int = 128
    d_rel: int = 64
    d_typ: int = 64
    query_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # v10+
    
    # Updater
    theta_match: float = 0.85
    theta_conflict: float = 0.3
    ema_alpha: float = 0.9  # v11+
    
    # SSM
    ssm_hidden_dim: int = 256
    
    # Init
    init_std: float = 0.02
    
    def small_test(self) -> DCortexConfig:
        """Smaller config for local testing."""
        # Returns config with vocab_size=256, hidden_dim=128, etc.
```

---

## Usage Patterns

### Pattern 1: Structural Episode Training

```python
def run_structural_episode(model, ep):
    model.reset_memory()
    model.begin_episode()
    
    fact_keys = []
    for fact in ep.facts:
        xf = tokenize(fact.text)
        ans_id = torch.tensor([fact.answer_token_id])
        aux = model.encode(
            xf,
            answer_token_id=ans_id,
            lexical_alpha=0.9,
            force_bank='working',
        )
        fact_keys.append(aux['w_k_ent'][0])
    
    xp = tokenize(ep.prompt)
    logits, retrieved = model.decode(xp, return_retrieved=True)
    aux_logits = model.aux_answer_head(retrieved)
    
    # Loss computation
    L_emit = F.cross_entropy(aux_logits, target)
    L_sel = compute_l_sel(fact_keys, q_k_ent, target_fact_idx)
    L_sep = compute_l_sep_neg(fact_keys)
    
    total = L_emit + L_sel + 0.5 * L_sep
    total.backward()
    model.clear_overlays()
```

### Pattern 2: Inference

```python
model.eval()
with torch.no_grad():
    model.reset_memory()
    
    for fact in facts:
        xf = tokenize(fact.text)
        ans_id = torch.tensor([fact.answer_token_id])
        model.encode(xf, answer_token_id=ans_id, 
                     lexical_alpha=0.9, force_bank='working')
    
    xp = tokenize(question)
    _, retrieved = model.decode(xp, return_retrieved=True)
    aux_logits = model.aux_answer_head(retrieved)
    predicted = aux_logits[0].argmax().item()
```

### Pattern 3: Language Modeling (No Memory)

```python
model.reset_memory()  # empty memory
logits = model.decode(batch_inputs)
lm_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
```

Used for parallel LM training to preserve language capability.

---

## Loss Functions

All loss functions are computed at the training script level (not methods of the model). See `colab/step2_training_v6.py` (v10) and `colab/step2_training_v11.py` (v11) for reference implementations.

**L_emit** (primary):
```python
L_emit = F.cross_entropy(
    aux_answer_head(retrieved_value),
    target_answer_token
)
```

**L_sel** (attention sharpness):
```python
K = F.normalize(torch.stack(fact_keys), dim=-1)
q_n = F.normalize(q_k_ent, dim=-1)
sim = (q_n @ K.T).squeeze(0)
log_p = F.log_softmax(sim * 5.0, dim=-1)  # temp 5.0 for supervision
L_sel = -log_p[target_fact_idx]
```

**L_sep_neg** (key separation):
```python
K_n = F.normalize(torch.stack(fact_keys), dim=-1)
sims = K_n @ K_n.T
mask = torch.eye(len(fact_keys), dtype=torch.bool)
off_diag = sims[~mask]
L_sep_neg = F.relu(off_diag - 0.5).pow(2).mean()
```

**L_occ** (occupancy, optional):
```python
L_occ = ((n_distinct_slots / n_distinct_entities) - 1.0) ** 2
```

v11 uses only `L_emit + L_sel + 0.5 * L_sep_neg`.

---

## Utilities

### verify_integration

**Location**: `scripts/verify_integration.py`

Parses the main training script, classifies modules as WIRED/NOT_WIRED/DUPLICATE. Returns non-zero exit code if critical modules not wired.

### test_forward_smoke

**Location**: `tests/test_forward_smoke.py`

End-to-end smoke test: encode facts, decode question, verify gradient flow through memory.

---

**End of API Reference**
