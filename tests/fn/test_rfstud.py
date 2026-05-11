"""Tests for rfstud."""
import numpy as np
import pytest
from morie.fn.rfstud import rfstud


def test_rfstud_basic():
    result = rfstud()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-StudentT"


def test_rfstud_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfstud(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfstud_no_data():
    result = rfstud(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfstud_alias():
    from morie.fn.rfstud import rfstud
    assert rfstud is rfstud
