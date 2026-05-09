"""Tests for cdcol."""
import numpy as np
import pytest
from moirais.fn.cdcol import cdcol


def test_cdcol_basic():
    result = cdcol()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "CoSim-ColocatedCK"


def test_cdcol_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = cdcol(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_cdcol_no_data():
    result = cdcol(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_cdcol_alias():
    from moirais.fn.cdcol import cdcol
    assert cdcol is cdcol
