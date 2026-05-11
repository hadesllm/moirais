"""Tests for glmmre."""
import numpy as np
import pytest
from morie.fn.glmmre import glmmre


def test_glmmre_basic():
    result = glmmre()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "GLMM-RandomEffects"


def test_glmmre_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = glmmre(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_glmmre_no_data():
    result = glmmre(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_glmmre_alias():
    from morie.fn.glmmre import glmmre
    assert glmmre is glmmre
