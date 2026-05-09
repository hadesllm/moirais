"""Tests for vdvor3."""
import numpy as np
import pytest
from moirais.fn.vdvor3 import vdvor3


def test_vdvor3_basic():
    result = vdvor3()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-3D"


def test_vdvor3_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdvor3(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdvor3_no_data():
    result = vdvor3(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdvor3_alias():
    from moirais.fn.vdvor3 import vdvor3
    assert vdvor3 is vdvor3
