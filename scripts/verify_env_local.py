# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha local bring-up.
# Gate 1 (ENV): verify CUDA availability, compute capability (12, 0) for the
# RTX 5080 (Blackwell sm_120), and that a real cuda matmul executes. Emits a
# single verdict line consumable by the training harness.

import json
import sys
from typing import Dict, Any

# Windows console is cp1252 by default; force utf-8 so status glyphs print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - older streams without reconfigure
        pass

import torch


SEP: str = "=" * 70


def verify_environment() -> Dict[str, Any]:
    """Run the ENV gate and return a structured result.

    Returns:
        Dict with keys: passed (bool), evidence (str), details (dict).
    """
    details: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    print(SEP, flush=True)
    print("[INFO] Gate 1 ENV verification", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] torch={details['torch_version']}  "
          f"cuda={details['torch_cuda_version']}  "
          f"available={details['cuda_available']}", flush=True)

    if not details["cuda_available"]:
        msg = "CUDA not available. torch build is CPU-only or driver is missing."
        print(f"[ERROR] {msg}", flush=True)
        return {"passed": False, "evidence": msg, "details": details}

    device_name: str = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_vram_gb: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    details["device_name"] = device_name
    details["capability"] = list(capability)
    details["total_vram_gb"] = round(total_vram_gb, 2)

    print(f"[INFO] GPU: {device_name}", flush=True)
    print(f"[INFO] compute capability: {capability[0]}.{capability[1]}", flush=True)
    print(f"[INFO] total VRAM: {total_vram_gb:.2f} GB", flush=True)

    if tuple(capability) != (12, 0):
        msg = (f"Expected compute capability (12, 0) for RTX 5080, "
               f"got {capability[0]}.{capability[1]}.")
        print(f"[ERROR] {msg}", flush=True)
        return {"passed": False, "evidence": msg, "details": details}

    # Real cuda matmul: build two large tensors on device, multiply, force a
    # device sync, and read back a finite scalar. A broken sm_120 kernel image
    # raises here with 'no kernel image available for execution on the device'.
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        checksum: float = float(c.float().sum().item())
        ok_finite: bool = checksum == checksum and abs(checksum) != float("inf")
        details["matmul_checksum"] = checksum
        details["matmul_finite"] = ok_finite
        print(f"[INFO] cuda bf16 matmul 4096x4096 OK, checksum={checksum:.1f}", flush=True)
    except Exception as exc:  # noqa: BLE001 - report the exact failure
        msg = f"cuda matmul failed: {type(exc).__name__}: {exc}"
        print(f"[ERROR] {msg}", flush=True)
        return {"passed": False, "evidence": msg, "details": details}

    if not details["matmul_finite"]:
        msg = "cuda matmul produced a non-finite checksum."
        print(f"[ERROR] {msg}", flush=True)
        return {"passed": False, "evidence": msg, "details": details}

    evidence = (f"cuda available; {device_name}; capability "
                f"{capability[0]}.{capability[1]}; bf16 4096x4096 matmul OK "
                f"(checksum={checksum:.1f}); torch={details['torch_version']} "
                f"cu={details['torch_cuda_version']}")
    print(f"✓ ENV gate PASS: {evidence}", flush=True)
    print(SEP, flush=True)
    return {"passed": True, "evidence": evidence, "details": details}


def main() -> int:
    result = verify_environment()
    print("ENV_VERDICT_JSON " + json.dumps(
        {"criterion_id": "ENV", "passed": result["passed"],
         "evidence": result["evidence"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
