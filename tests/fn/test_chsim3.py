"""Tests for chsim3."""
import numpy as np
import pytest
from morie.fn.chsim3 import chsim3


def test_chsim3_basic():
    result = chsim3()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyConditional"


def test_chsim3_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chsim3(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chsim3_no_data():
    result = chsim3(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chsim3_alias():
    from morie.fn.chsim3 import chsim3
    assert chsim3 is chsim3
