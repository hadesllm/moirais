"""Tests for glmmpred."""
import numpy as np
import pytest
from morie.fn.glmmpred import glmmpred


def test_glmmpred_basic():
    result = glmmpred()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-PosteriorPred"


def test_glmmpred_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmpred(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmpred_no_data():
    result = glmmpred(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmpred_alias():
    from morie.fn.glmmpred import glmmpred
    assert glmmpred is glmmpred
