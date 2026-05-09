"""Tests for chlexp."""
import numpy as np
import pytest
from moirais.fn.chlexp import chlexp


def test_chlexp_basic():
    result = chlexp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Cholesky-Exponential"


def test_chlexp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlexp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlexp_no_data():
    result = chlexp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlexp_alias():
    from moirais.fn.chlexp import chlexp
    assert chlexp is chlexp
