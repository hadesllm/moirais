"""Tests for fn/pstrat.py -- Post-stratification weights."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.pstrat import pstrat, poststratification_weights


def test_pstrat_returns_series():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "stratum": rng.choice(["A", "B", "C"], size=100),
        "y": rng.normal(0, 1, size=100),
    })
    pop_counts = {"A": 5000, "B": 3000, "C": 2000}
    result = pstrat(df, "stratum", pop_counts)
    assert isinstance(result, pd.Series)
    assert len(result) == 100


def test_pstrat_equal_distribution():
    """If sample proportions match population, weights should be ~1."""
    df = pd.DataFrame({"stratum": ["A"] * 50 + ["B"] * 50})
    pop_counts = {"A": 500, "B": 500}
    result = poststratification_weights(df, "stratum", pop_counts)
    assert np.allclose(result.values, 1.0)


def test_pstrat_missing_stratum():
    df = pd.DataFrame({"stratum": ["A", "B", "C"]})
    pop_counts = {"A": 100, "B": 200}  # missing C
    with pytest.raises(ValueError, match="missing"):
        pstrat(df, "stratum", pop_counts)
