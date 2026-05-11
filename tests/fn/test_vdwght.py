"""Tests for vdwght."""
import numpy as np
import pytest
from morie.fn.vdwght import vdwght


def test_vdwght_basic():
    result = vdwght()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-Weighted"


def test_vdwght_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdwght(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdwght_no_data():
    result = vdwght(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdwght_alias():
    from morie.fn.vdwght import vdwght
    assert vdwght is vdwght
