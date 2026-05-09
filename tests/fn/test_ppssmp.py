"""Tests for fn/ppssmp.py -- PPS sampling."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.ppssmp import ppssmp, pps_sample


def test_ppssmp_correct_size():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "size": rng.uniform(10, 1000, size=50),
        "name": [f"unit_{i}" for i in range(50)],
    })
    result = ppssmp(df, "size", 10, seed=42)
    assert len(result) == 10


def test_ppssmp_larger_units_more_likely():
    """Units with larger sizes should appear more often across many draws."""
    df = pd.DataFrame({
        "size": [1, 1, 1, 1, 1000],
        "id": range(5),
    })
    # Over many single draws, unit 4 (size=1000) should dominate
    counts = np.zeros(5)
    for seed in range(100):
        result = pps_sample(df, "size", 1, seed=seed)
        counts[result["id"].iloc[0]] += 1
    assert counts[4] > 50  # Should be selected most of the time


def test_ppssmp_no_positive():
    df = pd.DataFrame({"size": [0, -1, -5], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="No positive"):
        ppssmp(df, "size", 1, seed=42)
