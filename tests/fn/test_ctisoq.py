"""Tests for ctisoq."""
import numpy as np
import pytest
from moirais.fn.ctisoq import ctisoq


def test_ctisoq_basic():
    result = ctisoq()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Isoline"


def test_ctisoq_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctisoq(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctisoq_no_data():
    result = ctisoq(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctisoq_alias():
    from moirais.fn.ctisoq import ctisoq
    assert ctisoq is ctisoq
