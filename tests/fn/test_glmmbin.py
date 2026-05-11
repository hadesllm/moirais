"""Tests for glmmbin."""
import numpy as np
import pytest
from morie.fn.glmmbin import glmmbin


def test_glmmbin_basic():
    result = glmmbin()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-Spatial-Binomial"


def test_glmmbin_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmbin(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmbin_no_data():
    result = glmmbin(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmbin_alias():
    from morie.fn.glmmbin import glmmbin
    assert glmmbin is glmmbin
