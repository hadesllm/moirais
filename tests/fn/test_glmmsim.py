"""Tests for glmmsim."""
import numpy as np
import pytest
from moirais.fn.glmmsim import glmmsim


def test_glmmsim_basic():
    result = glmmsim()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-Spatial-Gaussian"


def test_glmmsim_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmsim(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmsim_no_data():
    result = glmmsim(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmsim_alias():
    from moirais.fn.glmmsim import glmmsim
    assert glmmsim is glmmsim
