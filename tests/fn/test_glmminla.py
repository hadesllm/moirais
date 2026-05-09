"""Tests for glmminla."""
import numpy as np
import pytest
from moirais.fn.glmminla import glmminla


def test_glmminla_basic():
    result = glmminla()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-INLA"


def test_glmminla_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmminla(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmminla_no_data():
    result = glmminla(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmminla_alias():
    from moirais.fn.glmminla import glmminla
    assert glmminla is glmminla
