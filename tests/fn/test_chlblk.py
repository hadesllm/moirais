"""Tests for chlblk."""
import numpy as np
import pytest
from morie.fn.chlblk import chlblk


def test_chlblk_basic():
    result = chlblk()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyBlock"


def test_chlblk_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chlblk(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chlblk_no_data():
    result = chlblk(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chlblk_alias():
    from morie.fn.chlblk import chlblk
    assert chlblk is chlblk
