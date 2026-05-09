"""Tests for glmmgam."""
import numpy as np
import pytest
from moirais.fn.glmmgam import glmmgam


def test_glmmgam_basic():
    result = glmmgam()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-Spatial-Gamma"


def test_glmmgam_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmgam(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmgam_no_data():
    result = glmmgam(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmgam_alias():
    from moirais.fn.glmmgam import glmmgam
    assert glmmgam is glmmgam
