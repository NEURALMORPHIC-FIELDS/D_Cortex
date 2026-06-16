# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Loads the SEALED v15.7a symbolic substrate (parse_fact, RoleOfModifierResolver,
# DeterministicObjectBank, ProvisionalMemory, BankStabilityIndex, CommitArbiterPas7a,
# the Pas7a consolidator pipeline, and the F1/F3/F5 holdout generators + vocabulary)
# from steps/13_v15_7a_consolidation/code.py WITHOUT editing it. The file is a Colab
# bundle whose module top level mounts Google Drive, asserts CUDA, and runs training
# and data preparation; it cannot be imported directly. This loader AST-parses the
# file and executes each top-level node in isolation, tolerating the nodes that fail
# (Colab imports, data-prep calls, training loops), so only the definitions and pure
# constants bind. The sealed file's bytes (and its sealed SHA) are never modified.

import ast
import contextlib
import io
import sys
import types
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SEALED_PATH = REPO_ROOT / "steps" / "13_v15_7a_consolidation" / "code.py"
_MODULE_NAME = "_dcortex_sealed_v15_7a"

# names the substrate top level expects (its own import nodes are skipped because the
# Colab/CUDA block sits among them); pre-binding them lets dataclass/typing resolve.
_PRELUDE = (
    "import torch, math, re, json, random, collections, dataclasses, enum, typing, "
    "itertools, functools, copy, hashlib, numpy as np, abc, warnings, time, os\n"
    "import torch.nn as nn, torch.nn.functional as F\n"
    "import numpy\n"
    "from dataclasses import dataclass, field\n"
    "from typing import (Optional, List, Dict, Tuple, Set, FrozenSet, Any, Callable, "
    "Union, Iterable, Sequence, Mapping, DefaultDict, NamedTuple)\n"
    "from enum import Enum, IntEnum, auto\n"
    "from collections import defaultdict, OrderedDict, deque, Counter, namedtuple\n"
)

# substrate symbols the integration spine depends on; load fails loudly if any miss.
REQUIRED = (
    "parse_fact", "parse_query", "RoleOfModifierResolver",
    "ProvisionalMemory", "ProvisionalEntry", "DeterministicObjectBank", "BankStabilityIndex",
    "CommitArbiterPas7a", "_v15_7a_run_consolidator_pipeline",
    "gen_F1_novel_paraphrase_syntax", "gen_F3_novel_lexical_alias", "gen_F5_novel_query_forms",
    "EXTERNAL_HOLDOUT_FAMILIES", "V15_ATTR_VALUES", "V15_1_ATTR_KEYWORDS", "HOLDOUT_ENTITIES_SINGLE",
)

_CACHE: Dict[str, object] = {}


def load_sealed_substrate(verbose: bool = False) -> Dict[str, object]:
    """Return the sealed substrate namespace (cached). The on-disk file is read only."""
    if _CACHE:
        return _CACHE
    if not SEALED_PATH.exists():
        raise FileNotFoundError(f"sealed substrate missing: {SEALED_PATH}")
    source = SEALED_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    module = types.ModuleType(_MODULE_NAME)
    sys.modules[_MODULE_NAME] = module          # required so dataclass forward-refs resolve
    ns = module.__dict__
    exec(compile(_PRELUDE, "<prelude>", "exec"), ns)

    nodes: List[ast.stmt] = list(tree.body)
    pending = list(range(len(nodes)))
    for _ in range(8):                          # multi-pass to resolve forward references
        still: List[int] = []
        for i in pending:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.Module(body=[nodes[i]], type_ignores=[]), str(SEALED_PATH), "exec"), ns)
            except Exception:  # noqa: BLE001 - Colab/CUDA/data-prep nodes are expected to fail
                still.append(i)
        if len(still) == len(pending):
            break
        pending = still

    missing = [name for name in REQUIRED if name not in ns]
    if missing:
        raise RuntimeError(f"sealed substrate load incomplete; missing {missing} "
                           f"({len(pending)} top-level nodes unresolved)")
    if verbose:
        print(f"[INFO] sealed v15.7a substrate loaded: {len(REQUIRED)} required symbols present, "
              f"{len(pending)} top-level nodes skipped (Colab/CUDA/data-prep)", flush=True)
    _CACHE.update(ns)
    return ns
