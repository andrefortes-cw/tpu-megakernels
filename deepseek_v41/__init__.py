"""DeepSeek-V4.1-Flash package bridge exposing the `deepseek_v41/tpu-v6e/` implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_SUBDIR = Path(__file__).resolve().parent / "tpu-v6e"
if str(_SUBDIR) not in sys.path:
    sys.path.insert(0, str(_SUBDIR))

_SPEC = importlib.util.spec_from_file_location(
    "deepseek_v41.tpu_v6e",
    _SUBDIR / "__init__.py",
    submodule_search_locations=[str(_SUBDIR)],
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load DeepSeek-V4.1-Flash TPU v6e module from {_SUBDIR}")

tpu_v6e = importlib.util.module_from_spec(_SPEC)
sys.modules["deepseek_v41.tpu_v6e"] = tpu_v6e
_SPEC.loader.exec_module(tpu_v6e)

DSV41Config = tpu_v6e.DSV41Config
DSV41JaxEngine = tpu_v6e.DSV41JaxEngine
DSV41PallasMegakernel = tpu_v6e.DSV41PallasMegakernel
DSparkDraftEngine = tpu_v6e.DSparkDraftEngine
load_dsv41_weights = tpu_v6e.load_dsv41_weights
load_dspark_weights = tpu_v6e.load_dspark_weights

__all__ = [
    "DSV41Config",
    "DSV41JaxEngine",
    "DSV41PallasMegakernel",
    "DSparkDraftEngine",
    "load_dspark_weights",
    "load_dsv41_weights",
    "tpu_v6e",
]
