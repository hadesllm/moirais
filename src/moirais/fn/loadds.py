# moirais.fn — function file (hadesllm/moirais)
"""Load dataset from CSV, TSV, Excel, Parquet, or JSON."""

from moirais.dataset import load_dataset as _fn

loadds = _fn
load_dataset = _fn


def cheatsheet() -> str:
    return "loadds() -> Load dataset from CSV, TSV, Excel, Parquet, or JSON."
