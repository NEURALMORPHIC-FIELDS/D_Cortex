"""D_Cortex v2.0-alpha — forward pass smoke test.

Verifies:
  * DCortexV2Model instantiates with the small_test config.
  * forward() returns logits of shape [B, T, vocab_size].
  * Memory banks receive writes over multiple forward passes.
  * reset_memory() clears all banks.
  * consolidate() runs without error.
  * A simple backward pass does not crash (gradients flow).
"""

import sys
from pathlib import Path

# Make the package importable when running this file directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model


def test_forward_smoke() -> None:
    torch.manual_seed(0)

    cfg = DCortexConfig().small_test()
    model = DCortexV2Model(cfg)
    model.eval()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    # --- Single forward ---
    logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"expected ({B}, {T}, {cfg.vocab_size}), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all().item(), "logits contain non-finite values"
    print(f"✓ forward OK: logits shape = {tuple(logits.shape)}")

    # --- Memory gets written over multiple passes ---
    before = model.memory_snapshot()
    for _ in range(5):
        _ = model(torch.randint(0, cfg.vocab_size, (B, T)))
    after = model.memory_snapshot()

    total_writes = sum(a["occupied"] for a in after.values())
    assert total_writes > 0, (
        f"expected at least one memory bank to be populated after 5 forwards, "
        f"got {after}"
    )
    print(f"✓ memory writes observed: "
          f"state={after['state']['occupied']}/{cfg.n_state_slots}  "
          f"episode_obj={after['episode_obj']['occupied']}/{cfg.n_episode_obj_slots}  "
          f"conflict={after['conflict']['occupied']}/{cfg.n_conflict_slots}  "
          f"archive={after['archive']['occupied']}/{cfg.n_archive_slots}  "
          f"working={after['working']['occupied']}/{cfg.n_work_slots}")

    # --- reset_memory clears everything ---
    model.reset_memory()
    cleared = model.memory_snapshot()
    cleared_total = sum(a["occupied"] for a in cleared.values())
    assert cleared_total == 0, f"after reset, expected 0 occupied slots, got {cleared_total}"
    print("✓ reset_memory clears all banks")

    # --- Re-populate then consolidate ---
    for _ in range(8):
        _ = model(torch.randint(0, cfg.vocab_size, (B, T)))
    report = model.consolidate()
    assert isinstance(report, dict), "consolidate() must return a dict"
    print("✓ consolidate runs without error")

    # --- Backward pass through logits ---
    model.train()
    model.reset_memory()
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
    )
    loss.backward()
    grad_norm = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().pow(2).sum().item())
            n_with_grad += 1
    grad_norm = grad_norm ** 0.5
    assert n_with_grad > 0, "no parameter received a gradient"
    assert torch.isfinite(torch.tensor(grad_norm)).item(), "grad norm is non-finite"
    print(f"✓ backward OK: loss={loss.item():.4f}  grad_norm={grad_norm:.4f}  "
          f"params_with_grad={n_with_grad}")

    print("\n" + "=" * 70)
    print("[INFO] smoke test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_forward_smoke()
