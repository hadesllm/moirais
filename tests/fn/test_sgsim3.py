"""Tests for sgsim3."""
import numpy as np
import pytest
from moirais.fn.sgsim3 import sgsim3


def test_sgsim3_basic():
    result = sgsim3()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SGS-OrdinaryKriging"


def test_sgsim3_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sgsim3(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sgsim3_no_data():
    result = sgsim3(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sgsim3_alias():
    from moirais.fn.sgsim3 import sgsim3
    assert sgsim3 is sgsim3
