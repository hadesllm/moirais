"""Tests for fn/clsmp.py -- Cluster sampling."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.clsmp import clsmp, cluster_sample


def test_clsmp_selects_correct_clusters():
    rng = np.random.default_rng(42)
    clusters = np.repeat(np.arange(10), 20)
    df = pd.DataFrame({
        "cluster": clusters,
        "y": rng.normal(0, 1, size=200),
    })
    result = clsmp(df, "cluster", 3, seed=42)
    assert result["cluster"].nunique() == 3


def test_clsmp_all_units_selected():
    """All units within selected clusters should be included."""
    df = pd.DataFrame({
        "cluster": [1] * 5 + [2] * 5 + [3] * 5,
        "y": range(15),
    })
    result = cluster_sample(df, "cluster", 2, seed=42)
    for c in result["cluster"].unique():
        original_count = (df["cluster"] == c).sum()
        sampled_count = (result["cluster"] == c).sum()
        assert original_count == sampled_count


def test_clsmp_too_many_clusters():
    df = pd.DataFrame({"cluster": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        clsmp(df, "cluster", 5, seed=42)
