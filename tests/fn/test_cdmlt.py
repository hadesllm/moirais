"""Tests for cdmlt."""
import numpy as np
import pytest
from moirais.fn.cdmlt import cdmlt


def test_cdmlt_basic():
    result = cdmlt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ConditionalSim-Multivar"


def test_cdmlt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdmlt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdmlt_no_data():
    result = cdmlt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdmlt_alias():
    from moirais.fn.cdmlt import cdmlt
    assert cdmlt is cdmlt
