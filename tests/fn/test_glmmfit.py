"""Tests for glmmfit."""
import numpy as np
import pytest
from moirais.fn.glmmfit import glmmfit


def test_glmmfit_basic():
    result = glmmfit()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-INLAFit"


def test_glmmfit_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmfit(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmfit_no_data():
    result = glmmfit(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmfit_alias():
    from moirais.fn.glmmfit import glmmfit
    assert glmmfit is glmmfit
