"""Tests for vdvor."""
import numpy as np
import pytest
from moirais.fn.vdvor import vdvor


def test_vdvor_basic():
    result = vdvor()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-2D"


def test_vdvor_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdvor(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdvor_no_data():
    result = vdvor(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdvor_alias():
    from moirais.fn.vdvor import vdvor
    assert vdvor is vdvor
