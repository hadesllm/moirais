"""Tests for tbexp."""
import numpy as np
import pytest
from moirais.fn.tbexp import tbexp


def test_tbexp_basic():
    result = tbexp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "TurningBands-Exponential"


def test_tbexp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = tbexp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_tbexp_no_data():
    result = tbexp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_tbexp_alias():
    from moirais.fn.tbexp import tbexp
    assert tbexp is tbexp
