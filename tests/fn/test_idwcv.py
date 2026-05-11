"""Tests for idwcv."""
import numpy as np
import pytest
from morie.fn.idwcv import idwcv


def test_idwcv_basic():
    result = idwcv()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-CrossValidation"


def test_idwcv_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwcv(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwcv_no_data():
    result = idwcv(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwcv_alias():
    from morie.fn.idwcv import idwcv
    assert idwcv is idwcv
