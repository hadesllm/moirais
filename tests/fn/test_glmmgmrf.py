"""Tests for glmmgmrf."""
import numpy as np
import pytest
from morie.fn.glmmgmrf import glmmgmrf


def test_glmmgmrf_basic():
    result = glmmgmrf()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-GMRF"


def test_glmmgmrf_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmgmrf(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmgmrf_no_data():
    result = glmmgmrf(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmgmrf_alias():
    from morie.fn.glmmgmrf import glmmgmrf
    assert glmmgmrf is glmmgmrf
