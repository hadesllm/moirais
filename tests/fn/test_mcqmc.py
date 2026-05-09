"""Tests for mcqmc."""
import numpy as np
import pytest
from moirais.fn.mcqmc import mcqmc


def test_mcqmc_basic():
    result = mcqmc()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-QuasiMC"


def test_mcqmc_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcqmc(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcqmc_no_data():
    result = mcqmc(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcqmc_alias():
    from moirais.fn.mcqmc import mcqmc
    assert mcqmc is mcqmc
