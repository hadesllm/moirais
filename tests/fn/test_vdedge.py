"""Tests for vdedge."""
import numpy as np
import pytest
from moirais.fn.vdedge import vdedge


def test_vdedge_basic():
    result = vdedge()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-EdgeLength"


def test_vdedge_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdedge(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdedge_no_data():
    result = vdedge(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdedge_alias():
    from moirais.fn.vdedge import vdedge
    assert vdedge is vdedge
