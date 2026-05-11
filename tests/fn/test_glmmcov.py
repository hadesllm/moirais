"""Tests for glmmcov."""
import numpy as np
import pytest
from morie.fn.glmmcov import glmmcov


def test_glmmcov_basic():
    result = glmmcov()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-CovParams"


def test_glmmcov_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmcov(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmcov_no_data():
    result = glmmcov(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmcov_alias():
    from morie.fn.glmmcov import glmmcov
    assert glmmcov is glmmcov
