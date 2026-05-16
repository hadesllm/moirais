"""Shared test fixtures — generates synthetic CPADS data for CI.

Real CPADS PUMF microdata lives in data/private/ and is never committed.
These fixtures create a synthetic dataset with canonical MORIE columns
so module tests run on CI without private data.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
import sys

# Keep the suite hermetic: morie's import-time update check must not
# reach PyPI or print notices during tests.
os.environ.setdefault("MORIE_NO_UPDATE_CHECK", "1")

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PY_PACKAGE = ROOT / "tools" / "py-package"

if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))


def _generate_synthetic_cpads(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic DataFrame with canonical CPADS columns."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    age_groups = ["15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    genders = ["male", "female", "non-binary/other"]
    provinces = ["BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL"]
    health = ["excellent", "very good", "good", "fair", "poor"]

    alcohol = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    heavy = np.where((alcohol == 1) & (rng.random(n) < 0.35), 1, 0)
    ebac_tot = np.where(
        alcohol == 1, np.round(rng.uniform(0.0, 0.18, size=n), 4), 0.0,
    )
    ebac_legal = (ebac_tot >= 0.08).astype(int)

    return pd.DataFrame({
        "weight": np.round(rng.uniform(0.3, 3.5, size=n), 6),
        "alcohol_past12m": alcohol,
        "heavy_drinking_30d": heavy,
        "ebac_tot": ebac_tot,
        "ebac_legal": ebac_legal,
        "cannabis_any_use": rng.choice([0, 1], size=n),
        "age_group": [random.choice(age_groups) for _ in range(n)],
        "gender": [random.choice(genders) for _ in range(n)],
        "province_region": [random.choice(provinces) for _ in range(n)],
        "mental_health": [random.choice(health) for _ in range(n)],
        "physical_health": [random.choice(health) for _ in range(n)],
    })


@pytest.fixture(scope="session")
def synthetic_cpads_csv(tmp_path_factory) -> Path:
    """Write a synthetic CPADS CSV and return its path."""
    df = _generate_synthetic_cpads()
    path = tmp_path_factory.mktemp("data") / "cpads-synthetic.csv"
    df.to_csv(path, index=False)
    return path


import ast
import inspect
import textwrap
import warnings

_TRIVIAL_PATTERNS = {
    "assert True",
    "assert result is not None",
    "assert r.value is not None",
    "assert r is not None",
    "assert result is result",
    "assert r is r",
    "assert r.name",
    "assert result is not None or",
    "assert r.value is not None or",
}


@pytest.fixture(autouse=True)
def _warn_trivial_assertions(request):
    """Flag tests that only use trivially-true assertions."""
    yield
    src = inspect.getsource(request.function)
    lines = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("assert")]
    if not lines:
        return
    trivial = sum(1 for ln in lines if any(ln.startswith(p) for p in _TRIVIAL_PATTERNS))
    if trivial == len(lines) and len(lines) > 0:
        warnings.warn(
            f"FALSE-POSITIVE RISK: {request.node.nodeid} has {trivial} trivially-true "
            f"assertion(s) and zero domain-specific checks. Tighten assertions.",
            stacklevel=2,
        )
