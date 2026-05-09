"""Tests for glmmnb."""
import numpy as np
import pytest
from moirais.fn.glmmnb import glmmnb


def test_glmmnb_basic():
    result = glmmnb()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-Spatial-NB"


def test_glmmnb_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmnb(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmnb_no_data():
    result = glmmnb(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmnb_alias():
    from moirais.fn.glmmnb import glmmnb
    assert glmmnb is glmmnb
