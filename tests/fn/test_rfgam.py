"""Tests for rfgam."""
import numpy as np
import pytest
from morie.fn.rfgam import rfgam


def test_rfgam_basic():
    result = rfgam()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Gamma"


def test_rfgam_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfgam(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfgam_no_data():
    result = rfgam(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfgam_alias():
    from morie.fn.rfgam import rfgam
    assert rfgam is rfgam
