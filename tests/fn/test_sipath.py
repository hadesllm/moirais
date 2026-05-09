"""Tests for sipath."""
import numpy as np
import pytest
from moirais.fn.sipath import sipath


def test_sipath_basic():
    result = sipath()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-Path"


def test_sipath_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sipath(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sipath_no_data():
    result = sipath(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sipath_alias():
    from moirais.fn.sipath import sipath
    assert sipath is sipath
