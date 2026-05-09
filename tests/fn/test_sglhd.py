"""Tests for sglhd."""
import numpy as np
import pytest
from moirais.fn.sglhd import sglhd


def test_sglhd_basic():
    result = sglhd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-LHS"


def test_sglhd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sglhd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sglhd_no_data():
    result = sglhd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sglhd_alias():
    from moirais.fn.sglhd import sglhd
    assert sglhd is sglhd
