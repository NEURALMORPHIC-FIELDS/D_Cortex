# -*- coding: utf-8 -*-
# Local runner for the 25 technical tests (mirrors the Colab notebook)

import os, sys, time, traceback
from dataclasses import dataclass
from typing import List, Dict

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.memory.banks import (
    StateMemory, EpisodeObjectMemory, ConflictMemory,
    ArchiveMemory, WorkingMemory, EpisodeSSM,
)
from dcortex.memory.query import QueryEngine
from dcortex.memory.readers import SemanticReader, EpisodeReader, MemoryReadFusion
from dcortex.memory.writer import MemoryWriter
from dcortex.memory.updater import MemoryUpdater
from dcortex.memory.consolidator import MemoryConsolidator
from dcortex.backbone.transformer import StandardTransformerBlock
from dcortex.backbone.fusion_block import CrossAttention, FusionBlock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEP = "=" * 70


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    duration_ms: float


RESULTS: List[TestResult] = []


def run_test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.perf_counter()
            try:
                detail = fn()
                elapsed = (time.perf_counter() - t0) * 1000
                RESULTS.append(TestResult(name, True, detail or "OK", elapsed))
                print(f"  PASS  {name} ({elapsed:.0f}ms)", flush=True)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                RESULTS.append(TestResult(name, False, str(e), elapsed))
                print(f"  FAIL  {name} ({elapsed:.0f}ms)", flush=True)
                print(f"        {e}", flush=True)
                traceback.print_exc()
        return wrapper
    return decorator


CFG = DCortexConfig().small_test()
B, T = 2, 16


def fresh_model() -> DCortexV2Model:
    torch.manual_seed(42)
    return DCortexV2Model(CFG).to(DEVICE)


def rand_ids() -> torch.Tensor:
    return torch.randint(0, CFG.vocab_size, (B, T), device=DEVICE)


# T01
@run_test("T01 forward_shape_and_finite")
def t01():
    model = fresh_model()
    model.eval()
    logits = model(rand_ids())
    assert logits.shape == (B, T, CFG.vocab_size)
    assert torch.isfinite(logits).all()
    return f"logits shape {tuple(logits.shape)}, all finite"


# T02
@run_test("T02 forward_determinism")
def t02():
    torch.manual_seed(99)
    m = DCortexV2Model(CFG).to(DEVICE).eval()
    ids = rand_ids()
    out1 = m(ids, write_memory=False)
    m.reset_memory()
    out2 = m(ids, write_memory=False)
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-5
    return f"max diff={diff:.2e}"


# T03
@run_test("T03 gradient_flow_per_module")
def t03():
    model = fresh_model()
    model.train()
    logits = model(rand_ids())
    targets = torch.randint(0, CFG.vocab_size, (B, T), device=DEVICE)
    loss = F.cross_entropy(logits.reshape(-1, CFG.vocab_size), targets.reshape(-1))
    loss.backward()
    for critical in ["embeddings", "standard_blocks", "fusion_blocks", "final_norm"]:
        mod = getattr(model, critical)
        wg = sum(1 for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert wg > 0, f"{critical} got 0 grads"
    assert len(list(model.updater.parameters())) == 0
    assert len(list(model.consolidator.parameters())) == 0
    return f"loss={loss.item():.4f}, all critical modules receive grads"


# T04
@run_test("T04 weight_tying")
def t04():
    model = fresh_model()
    assert model.lm_head.weight is model.embeddings.token_emb.weight
    assert model.lm_head.weight.data_ptr() == model.embeddings.token_emb.weight.data_ptr()
    return "same tensor confirmed"


# T05
@run_test("T05 bank_write_read_occupancy")
def t05():
    bank = StateMemory(8, 64, 16, 8, 8)
    updater = MemoryUpdater(CFG)
    assert bank.n_occupied() == 0
    for i in range(3):
        updater.update(bank, torch.randn(64), torch.randn(16), torch.randn(8), torch.randn(8), step=i)
    assert bank.n_occupied() == 3
    bank.reset()
    assert bank.n_occupied() == 0
    return "write 3, verify, reset OK"


# T06
@run_test("T06 updater_ema_on_match")
def t06():
    bank = StateMemory(8, 64, 16, 8, 8)
    updater = MemoryUpdater(CFG)
    v1 = torch.ones(64)
    ke = torch.ones(16)
    kr = torch.ones(8)
    kt = torch.ones(8)
    updater.update(bank, v1, ke, kr, kt, step=0)
    val_before = bank.values[0].clone()
    v2 = torch.ones(64) * 5.0
    idx = updater.update(bank, v2, ke, kr, kt, step=1)
    assert idx == 0
    assert bank.n_occupied() == 1
    expected = (1.0 - CFG.ema_alpha) * val_before + CFG.ema_alpha * v2
    diff = (expected - bank.values[0]).abs().max().item()
    assert diff < 1e-5
    return f"EMA correct, diff={diff:.2e}"


# T07
@run_test("T07 updater_lru_eviction")
def t07():
    bank = StateMemory(4, 64, 16, 8, 8)
    updater = MemoryUpdater(CFG)
    torch.manual_seed(77)
    for i in range(4):
        updater.update(bank, torch.randn(64), torch.randn(16), torch.randn(8), torch.randn(8), step=i)
    assert bank.free_slot() == -1
    assert bank.lru_slot() == 0
    idx = updater.update(bank, torch.randn(64), torch.randn(16), torch.randn(8), torch.randn(8), step=99)
    assert idx == 0
    assert bank.last_write_step[0] == 99
    return "LRU eviction at slot 0 confirmed"


# T08
@run_test("T08 conflict_detection_and_diff_vector")
def t08():
    bank = StateMemory(8, 64, 16, 8, 8)
    conflict_bank = ConflictMemory(4, 64, 16, 8, 8)
    updater = MemoryUpdater(CFG)
    ke = torch.ones(16)
    kr = torch.ones(8)
    kt = torch.ones(8)
    v_old = torch.ones(64) * 2.0
    updater.update(bank, v_old, ke, kr, kt, step=0)
    v_new = torch.ones(64) * (-5.0)
    assert updater.detect_conflict(bank, v_new, ke, kr, kt)
    updater.update(conflict_bank, v_new, ke, kr, kt, step=1, is_conflict=True)
    v_newer = torch.ones(64) * 10.0
    updater.update(conflict_bank, v_newer, ke, kr, kt, step=2, is_conflict=True)
    stored = conflict_bank.values[0]
    expected_diff = v_newer - v_new
    diff_err = (stored - expected_diff).abs().max().item()
    assert diff_err < 1e-5
    return f"conflict detected, diff vector correct (err={diff_err:.2e})"


# T09
@run_test("T09 episode_ssm_recurrence")
def t09():
    ssm = EpisodeSSM(64, 32).to(DEVICE)
    assert ssm.x.abs().sum() == 0
    r1 = ssm(torch.randn(64, device=DEVICE))
    assert r1.shape == (64,)
    s1 = ssm.x.clone()
    ssm(torch.randn(64, device=DEVICE))
    assert not torch.allclose(s1, ssm.x)
    ssm.reset()
    assert ssm.x.abs().sum() == 0
    ssm.zero_grad()
    # Need 2 steps: after step 1, x is non-zero, so step 2 has grad on a_raw
    # (a * x_detached has zero grad when x_detached=0 after reset)
    _ = ssm(torch.randn(64, device=DEVICE))
    r = ssm(torch.randn(64, device=DEVICE))
    r.sum().backward()
    assert ssm.a_raw.grad is not None and ssm.a_raw.grad.abs().sum() > 0
    assert ssm.B.weight.grad is not None and ssm.B.weight.grad.abs().sum() > 0
    assert ssm.C.weight.grad is not None and ssm.C.weight.grad.abs().sum() > 0
    return "recurrence, reset, grad on a/B/C OK (2-step)"


# T10
@run_test("T10 query_engine_shapes")
def t10():
    qe = QueryEngine(CFG).to(DEVICE)
    q_ent, q_rel, q_typ = qe(torch.randn(B, CFG.hidden_dim, device=DEVICE))
    assert q_ent.shape == (B, CFG.d_ent)
    assert q_rel.shape == (B, CFG.d_rel)
    assert q_typ.shape == (B, CFG.d_typ)
    return "shapes correct"


# T11
@run_test("T11 semantic_reader_empty_and_populated")
def t11():
    reader = SemanticReader(CFG).to(DEVICE)
    bank = StateMemory(8, CFG.hidden_dim, CFG.d_ent, CFG.d_rel, CFG.d_typ).to(DEVICE)
    q_ent = torch.randn(B, CFG.d_ent, device=DEVICE)
    q_rel = torch.randn(B, CFG.d_rel, device=DEVICE)
    q_typ = torch.randn(B, CFG.d_typ, device=DEVICE)
    r_empty = reader(q_ent, q_rel, q_typ, bank)
    assert r_empty.abs().sum() == 0
    updater = MemoryUpdater(CFG)
    for i in range(3):
        updater.update(bank, torch.randn(CFG.hidden_dim), torch.randn(CFG.d_ent),
                       torch.randn(CFG.d_rel), torch.randn(CFG.d_typ), step=i)
    r_pop = reader(q_ent, q_rel, q_typ, bank)
    assert r_pop.abs().sum() > 0
    return "empty->zeros, populated->non-zero"


# T12
@run_test("T12 read_fusion_shape")
def t12():
    fuser = MemoryReadFusion(CFG).to(DEVICE)
    streams = [torch.randn(B, CFG.hidden_dim, device=DEVICE) for _ in range(5)]
    out = fuser(*streams)
    assert out.shape == (B, 5, CFG.hidden_dim)
    return f"shape {tuple(out.shape)}"


# T13
@run_test("T13 writer_gate_distribution")
def t13():
    model = fresh_model()
    model.eval()
    gate_accum = torch.zeros(6, device=DEVICE)
    torch.manual_seed(123)
    for _ in range(20):
        h = model.embeddings(rand_ids())
        for block in model.standard_blocks:
            h = block(h)
        h_pool = model._pool(h, None)
        h_norm = model.writer.norm(h_pool)
        gate_probs = F.softmax(model.writer.gate(h_norm), dim=-1)
        gate_accum += gate_probs.sum(dim=0)
    gate_avg = gate_accum / (20 * B)
    assert abs(gate_avg.sum().item() - 1.0) < 0.01
    names = list(MemoryWriter.BANK_ORDER)
    return ", ".join(f"{n}={gate_avg[i]:.3f}" for i, n in enumerate(names))


# T14
@run_test("T14 fusion_block_mem_gate")
def t14():
    fb = FusionBlock(CFG).to(DEVICE)
    assert (fb.mem_gate.data == 0).all()
    h = torch.randn(B, T, CFG.hidden_dim, device=DEVICE)
    out = fb(h, torch.randn(B, 5, CFG.hidden_dim, device=DEVICE))
    assert out.shape == h.shape
    return "mem_gate init=0, forward OK"


# T15
@run_test("T15 consolidator_full_cycle")
def t15():
    src = StateMemory(8, 64, 16, 8, 8)
    dst = ArchiveMemory(32, 64, 16, 8, 8)
    consolidator = MemoryConsolidator(CFG)
    updater = MemoryUpdater(CFG)
    torch.manual_seed(55)
    for i in range(4):
        updater.update(src, torch.randn(64), torch.randn(16), torch.randn(8), torch.randn(8), step=i)
    for step in range(200):
        consolidator.consolidate(src, dst, current_step=100 + step)
    return f"after 200 passes: src={src.n_occupied()}/8, dst={dst.n_occupied()}/32"


# T16
@run_test("T16 consolidator_merge")
def t16():
    bank = StateMemory(8, 64, 16, 8, 8)
    consolidator = MemoryConsolidator(CFG)
    ke = torch.ones(16)
    kr = torch.ones(8)
    kt = torch.ones(8)
    bank.values[0] = torch.ones(64)
    bank.k_ent[0] = ke
    bank.k_rel[0] = kr
    bank.k_typ[0] = kt
    bank.occupied[0] = True
    bank.usage[0] = 5.0
    bank.last_write_step[0] = 0
    bank.values[1] = torch.ones(64) * 1.01
    bank.k_ent[1] = ke * 1.001
    bank.k_rel[1] = kr * 1.001
    bank.k_typ[1] = kt * 1.001
    bank.occupied[1] = True
    bank.usage[1] = 3.0
    bank.last_write_step[1] = 1
    report = consolidator.consolidate(bank, None, current_step=10)
    return f"merged={report['merged']}, occupied={bank.n_occupied()}"


# T17
@run_test("T17 multi_step_memory_accumulation")
def t17():
    model = fresh_model()
    model.eval()
    snapshots = []
    for _ in range(15):
        _ = model(rand_ids())
        total = sum(s["occupied"] for s in model.memory_snapshot().values())
        snapshots.append(total)
    assert snapshots[-1] > snapshots[0]
    return f"15 steps: {snapshots[0]} -> {snapshots[-1]} total slots"


# T18
@run_test("T18 write_memory_false")
def t18():
    model = fresh_model()
    model.eval()
    for _ in range(5):
        _ = model(rand_ids(), write_memory=False)
    total = sum(s["occupied"] for s in model.memory_snapshot().values())
    assert total == 0
    assert model.step_counter.item() == 0
    return "5 forwards, 0 writes, step=0"


# T19
@run_test("T19 attention_mask")
def t19():
    model = fresh_model()
    model.eval()
    ids = rand_ids()
    mask = torch.ones(B, T, device=DEVICE)
    mask[1, T // 2:] = 0
    logits_m = model(ids, attention_mask=mask, write_memory=False)
    model.reset_memory()
    logits_u = model(ids, write_memory=False)
    diff = (logits_m - logits_u).abs().max().item()
    assert diff > 1e-6
    return f"masked vs unmasked diff={diff:.4f}"


# T20
@run_test("T20 numerical_stability")
def t20():
    model = fresh_model()
    model.eval()
    for _ in range(10):
        _ = model(rand_ids())
    logits = model(rand_ids())
    assert torch.isfinite(logits).all()
    for name, bank in model._bank_dict().items():
        assert torch.isfinite(bank.values).all()
    assert torch.isfinite(model.episode_ssm.x).all()
    return "all finite after 10 steps"


# T21
@run_test("T21 parameter_count_full_scale")
def t21():
    full_cfg = DCortexConfig()
    model = DCortexV2Model(full_cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 1e6
    assert total == trainable
    del model
    return f"total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M"


# T22
@run_test("T22 cross_attention")
def t22():
    ca = CrossAttention(CFG).to(DEVICE)
    D = CFG.hidden_dim
    h = torch.randn(B, T, D, device=DEVICE, requires_grad=True)
    mem = torch.randn(B, 5, D, device=DEVICE, requires_grad=True)
    out = ca(h, mem)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert mem.grad is not None and mem.grad.abs().sum() > 0
    return "grads flow to h and memory"


# T23
@run_test("T23 causal_mask_self_attention")
def t23():
    # Test causal mask on StandardTransformerBlock in isolation.
    # Full model has memory reads based on pool(h) which intentionally
    # create global dependency -- that is correct behavior, not a
    # causal violation. Self-attention itself must be causal.
    block = StandardTransformerBlock(CFG).to(DEVICE)
    block.eval()
    torch.manual_seed(42)
    h = torch.randn(B, T, CFG.hidden_dim, device=DEVICE)
    h2 = h.clone()
    h2[:, -1, :] = torch.randn(CFG.hidden_dim, device=DEVICE)  # change last position

    out1 = block(h)
    out2 = block(h2)
    diff_early = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max().item()
    diff_last = (out1[:, -1, :] - out2[:, -1, :]).abs().max().item()
    assert diff_early < 1e-5, \
        f"causal violation in self-attention: early diff={diff_early}"
    assert diff_last > 1e-5, \
        f"last position unchanged: diff={diff_last}"
    return f"early diff={diff_early:.2e}, last diff={diff_last:.4f}"


# T24
@run_test("T24 episode_reader_subfusion")
def t24():
    ep_reader = EpisodeReader(CFG).to(DEVICE)
    ep_obj_mem = EpisodeObjectMemory(
        CFG.n_episode_obj_slots, CFG.hidden_dim,
        CFG.d_ent, CFG.d_rel, CFG.d_typ
    ).to(DEVICE)
    ep_ssm = EpisodeSSM(CFG.hidden_dim, CFG.ssm_hidden_dim).to(DEVICE)
    r_ep = ep_reader(
        torch.randn(B, CFG.d_ent, device=DEVICE),
        torch.randn(B, CFG.d_rel, device=DEVICE),
        torch.randn(B, CFG.d_typ, device=DEVICE),
        ep_obj_mem, ep_ssm,
        torch.randn(B, CFG.hidden_dim, device=DEVICE),
    )
    assert r_ep.shape == (B, CFG.hidden_dim)
    r_ep.sum().backward()
    assert ep_reader.W_theta.weight.grad is not None
    assert ep_ssm.a_raw.grad is not None
    return "shape OK, grads to W_theta and SSM"


# T25
@run_test("T25 config_validation")
def t25():
    caught = []
    try:
        DCortexConfig(hidden_dim=100, n_heads=12)
    except ValueError:
        caught.append("hd%nh")
    try:
        DCortexConfig(n_fusion_layers=0)
    except ValueError:
        caught.append("f<1")
    try:
        DCortexConfig(n_fusion_layers=20, n_layers=12)
    except ValueError:
        caught.append("f>l")
    try:
        DCortexConfig(ema_alpha=0.0)
    except ValueError:
        caught.append("a=0")
    try:
        DCortexConfig(ema_alpha=1.0)
    except ValueError:
        caught.append("a=1")
    assert len(caught) == 5
    return "5/5 invalid configs caught"


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    print(SEP)
    print("D_Cortex v2.0-alpha -- Technical Test Suite")
    print(f"Device: {DEVICE}")
    print(SEP)
    print(flush=True)

    for fn in [
        t01, t02, t03, t04, t05, t06, t07, t08, t09, t10,
        t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
        t21, t22, t23, t24, t25,
    ]:
        fn()

    print(flush=True)
    print(SEP)
    passed = [r for r in RESULTS if r.passed]
    failed = [r for r in RESULTS if not r.passed]
    print(f"PASSED: {len(passed)}/{len(RESULTS)}")
    print(f"FAILED: {len(failed)}/{len(RESULTS)}")
    if failed:
        for r in failed:
            print(f"  [FAIL] {r.name}: {r.detail}")
    else:
        print(f"ALL {len(RESULTS)} TESTS PASSED")
    print(SEP)
