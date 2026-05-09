"""Tests for cdcosm."""
import numpy as np
import pytest
from moirais.fn.cdcosm import cdcosm


def test_cdcosm_basic():
    result = cdcosm()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CoSim-LMC"


def test_cdcosm_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdcosm(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdcosm_no_data():
    result = cdcosm(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdcosm_alias():
    from moirais.fn.cdcosm import cdcosm
    assert cdcosm is cdcosm
