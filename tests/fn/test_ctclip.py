"""Tests for ctclip."""
import numpy as np
import pytest
from morie.fn.ctclip import ctclip


def test_ctclip_basic():
    result = ctclip()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Clip"


def test_ctclip_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctclip(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctclip_no_data():
    result = ctclip(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctclip_alias():
    from morie.fn.ctclip import ctclip
    assert ctclip is ctclip
