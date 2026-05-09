"""Tests for vdrefn."""
import numpy as np
import pytest
from moirais.fn.vdrefn import vdrefn


def test_vdrefn_basic():
    result = vdrefn()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Delaunay-Refinement"


def test_vdrefn_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = vdrefn(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_vdrefn_no_data():
    result = vdrefn(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_vdrefn_alias():
    from moirais.fn.vdrefn import vdrefn
    assert vdrefn is vdrefn
