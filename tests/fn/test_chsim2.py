"""Tests for chsim2."""
import numpy as np
import pytest
from moirais.fn.chsim2 import chsim2


def test_chsim2_basic():
    result = chsim2()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CholeskyNugget"


def test_chsim2_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = chsim2(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_chsim2_no_data():
    result = chsim2(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_chsim2_alias():
    from moirais.fn.chsim2 import chsim2
    assert chsim2 is chsim2
