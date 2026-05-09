"""Tests for fn/srs.py -- Simple random sample."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.srs import srs, simple_random_sample


def test_srs_correct_size():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, size=1000)})
    result = srs(df, 50, seed=42)
    assert len(result) == 50


def test_srs_reproducible():
    df = pd.DataFrame({"x": range(500)})
    s1 = simple_random_sample(df, 20, seed=42)
    s2 = simple_random_sample(df, 20, seed=42)
    assert s1["x"].tolist() == s2["x"].tolist()


def test_srs_no_replacement_exceeds():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        srs(df, 10, replace=False)


def test_srs_with_replacement():
    df = pd.DataFrame({"x": range(5)})
    result = srs(df, 20, replace=True, seed=42)
    assert len(result) == 20
