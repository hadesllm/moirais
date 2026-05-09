"""Tests for rbfphs."""
import numpy as np
import pytest
from moirais.fn.rbfphs import rbfphs


def test_rbfphs_basic():
    result = rbfphs()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RBF-Polyharmonic"


def test_rbfphs_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rbfphs(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rbfphs_no_data():
    result = rbfphs(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rbfphs_alias():
    from moirais.fn.rbfphs import rbfphs
    assert rbfphs is rbfphs
