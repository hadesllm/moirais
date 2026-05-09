"""Tests for mcvar."""
import numpy as np
import pytest
from moirais.fn.mcvar import mcvar


def test_mcvar_basic():
    result = mcvar()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-VarianceReduction"


def test_mcvar_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcvar(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcvar_no_data():
    result = mcvar(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcvar_alias():
    from moirais.fn.mcvar import mcvar
    assert mcvar is mcvar
