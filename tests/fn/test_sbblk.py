"""Tests for sbblk."""
import numpy as np
import pytest
from morie.fn.sbblk import sbblk


def test_sbblk_basic():
    result = sbblk()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Block"


def test_sbblk_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbblk(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbblk_no_data():
    result = sbblk(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbblk_alias():
    from morie.fn.sbblk import sbblk
    assert sbblk is sbblk
