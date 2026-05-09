"""Tests for sbcrc."""
import numpy as np
import pytest
from moirais.fn.sbcrc import sbcrc


def test_sbcrc_basic():
    result = sbcrc()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpatialBootstrap-Circular"


def test_sbcrc_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = sbcrc(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_sbcrc_no_data():
    result = sbcrc(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_sbcrc_alias():
    from moirais.fn.sbcrc import sbcrc
    assert sbcrc is sbcrc
