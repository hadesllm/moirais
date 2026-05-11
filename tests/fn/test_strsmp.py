"""Tests for fn/strsmp.py -- Stratified random sample."""
import numpy as np
import pandas as pd
import pytest

from morie.fn.strsmp import strsmp, stratified_sample


def test_strsmp_fixed_allocation():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "stratum": rng.choice(["A", "B"], size=200),
        "x": rng.normal(0, 1, size=200),
    })
    result = strsmp(df, "stratum", 10, seed=42)
    counts = result.groupby("stratum").size()
    assert all(counts == 10)


def test_strsmp_proportional():
    df = pd.DataFrame({
        "stratum": ["A"] * 80 + ["B"] * 20,
        "x": range(100),
    })
    result = stratified_sample(df, "stratum", 50, proportional=True, seed=42)
    assert len(result) == 50
    # A should have more samples than B
    counts = result.groupby("stratum").size()
    assert counts.get("A", 0) > counts.get("B", 0)


def test_strsmp_exceeds_stratum():
    df = pd.DataFrame({"stratum": ["A"] * 3 + ["B"] * 3, "x": range(6)})
    with pytest.raises(ValueError):
        strsmp(df, "stratum", 10, seed=42)
