# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Integration verifier. Parses dcortex/model.py, recursively collects every
# module reachable from DCortexV2Model.__init__, scans the rest of the
# dcortex/ tree, and classifies each .py file as WIRED / NOT_WIRED / DUPLICATE.
# Exits non-zero if any CRITICAL module is NOT_WIRED.

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "dcortex"

# Critical modules: if any of these is not wired, the verifier FAILS.
CRITICAL = {
    "dcortex.config",
    "dcortex.memory.banks",
    "dcortex.memory.query",
    "dcortex.memory.updater",
    "dcortex.memory.readers",
    "dcortex.memory.writer",
    "dcortex.memory.consolidator",
    "dcortex.backbone.embeddings",
    "dcortex.backbone.transformer",
    "dcortex.backbone.fusion_block",
    "dcortex.model",
}


def path_to_module(p: Path) -> str:
    rel = p.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def list_all_modules() -> List[str]:
    modules = []
    for root, _, files in os.walk(PKG_ROOT):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = Path(root) / f
            if f == "__init__.py":
                continue
            modules.append(path_to_module(p))
    return sorted(modules)


def extract_imports(path: Path) -> Set[str]:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] could not read {path}: {e}")
        return set()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        print(f"[ERROR] syntax error in {path}: {e}")
        return set()

    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            imports.add(node.module)
    return imports


def transitive_closure(entry_module: str) -> Set[str]:
    """BFS over internal (dcortex.*) imports starting at entry_module."""
    visited: Set[str] = set()
    frontier = [entry_module]
    while frontier:
        mod = frontier.pop()
        if mod in visited:
            continue
        visited.add(mod)
        path = REPO_ROOT / (mod.replace(".", "/") + ".py")
        if not path.exists():
            continue
        imports = extract_imports(path)
        for imp in imports:
            if imp.startswith("dcortex"):
                # Normalize: if importing a module directly
                if imp not in visited:
                    frontier.append(imp)
    return visited


def classify(all_modules: List[str], wired: Set[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for mod in all_modules:
        if mod in wired:
            result[mod] = "WIRED"
        else:
            result[mod] = "NOT_WIRED"
    # DUPLICATE detection: same filename in two different paths (by basename)
    basenames: Dict[str, List[str]] = {}
    for mod in all_modules:
        base = mod.split(".")[-1]
        basenames.setdefault(base, []).append(mod)
    for base, mods in basenames.items():
        if len(mods) > 1:
            for m in mods:
                result[m] = result[m] + "+DUPLICATE"
    return result


def report(result: Dict[str, str]) -> int:
    sep = "=" * 70
    print(sep)
    print("[INFO] D_Cortex v2.0-alpha :: integration verification")
    print(sep)

    wired = [m for m, s in result.items() if "WIRED" in s and "NOT_WIRED" not in s]
    not_wired = [m for m, s in result.items() if "NOT_WIRED" in s]
    duplicates = [m for m, s in result.items() if "DUPLICATE" in s]

    print(f"  total modules : {len(result)}")
    print(f"  WIRED         : {len(wired)}")
    print(f"  NOT_WIRED     : {len(not_wired)}")
    print(f"  DUPLICATE     : {len(duplicates)}")
    print("")
    print("Per-module status:")
    for mod in sorted(result):
        status = result[mod]
        critical_tag = " [CRITICAL]" if mod in CRITICAL else ""
        prefix = "✓" if "NOT_WIRED" not in status else "[ERROR]"
        print(f"  {prefix} {mod:55s} {status}{critical_tag}")

    # Fail if any CRITICAL is NOT_WIRED
    critical_unwired = [
        m for m in CRITICAL
        if m in result and "NOT_WIRED" in result[m]
    ]
    missing_critical = [m for m in CRITICAL if m not in result]

    print("")
    if critical_unwired:
        print("[ERROR] Critical modules NOT wired:")
        for m in critical_unwired:
            print(f"        - {m}")
        return 1
    if missing_critical:
        print("[ERROR] Critical modules missing from disk:")
        for m in missing_critical:
            print(f"        - {m}")
        return 1

    print("✓ All critical modules are WIRED.")
    print(sep)
    return 0


def main() -> int:
    if not PKG_ROOT.exists():
        print(f"[ERROR] package root not found: {PKG_ROOT}")
        return 2

    # Entry point: the model wrapper
    entry = "dcortex.model"
    wired = transitive_closure(entry)

    # Also always consider the entry itself wired
    wired.add(entry)

    all_modules = list_all_modules()
    classification = classify(all_modules, wired)
    return report(classification)


if __name__ == "__main__":
    sys.exit(main())
