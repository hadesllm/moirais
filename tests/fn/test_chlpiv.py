"""Tests for chlpiv."""
import numpy as np
import pytest
from morie.fn.chlpiv import chlpiv


def test_chlpiv_basic():
    result = chlpiv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "PivotedCholesky"


def test_chlpiv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlpiv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlpiv_no_data():
    result = chlpiv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlpiv_alias():
    from morie.fn.chlpiv import chlpiv
    assert chlpiv is chlpiv
