"""Tests for vdclip."""
import numpy as np
import pytest
from moirais.fn.vdclip import vdclip


def test_vdclip_basic():
    result = vdclip()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Voronoi-Clipping"


def test_vdclip_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdclip(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdclip_no_data():
    result = vdclip(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdclip_alias():
    from moirais.fn.vdclip import vdclip
    assert vdclip is vdclip
