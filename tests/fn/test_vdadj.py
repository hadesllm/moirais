"""Tests for vdadj."""
import numpy as np
import pytest
from moirais.fn.vdadj import vdadj


def test_vdadj_basic():
    result = vdadj()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-Adjacency"


def test_vdadj_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdadj(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdadj_no_data():
    result = vdadj(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdadj_alias():
    from moirais.fn.vdadj import vdadj
    assert vdadj is vdadj
