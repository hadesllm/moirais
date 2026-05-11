"""Tests for stiag."""
import numpy as np
import pytest
from morie.fn.stiag import stiag


def test_stiag_basic():
    result = stiag()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-IacoCesare"


def test_stiag_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stiag(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stiag_no_data():
    result = stiag(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stiag_alias():
    from morie.fn.stiag import stiag
    assert stiag is stiag
