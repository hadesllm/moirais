"""Tests for cdpost."""
import numpy as np
import pytest
from moirais.fn.cdpost import cdpost


def test_cdpost_basic():
    result = cdpost()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-Posterior"


def test_cdpost_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdpost(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdpost_no_data():
    result = cdpost(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdpost_alias():
    from moirais.fn.cdpost import cdpost
    assert cdpost is cdpost
