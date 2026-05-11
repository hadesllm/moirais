"""Tests for rfmix."""
import numpy as np
import pytest
from morie.fn.rfmix import rfmix


def test_rfmix_basic():
    result = rfmix()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Mixture"


def test_rfmix_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfmix(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfmix_no_data():
    result = rfmix(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfmix_alias():
    from morie.fn.rfmix import rfmix
    assert rfmix is rfmix
