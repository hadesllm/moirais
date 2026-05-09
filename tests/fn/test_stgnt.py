"""Tests for stgnt."""
import numpy as np
import pytest
from moirais.fn.stgnt import stgnt


def test_stgnt_basic():
    result = stgnt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Gneiting"


def test_stgnt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stgnt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stgnt_no_data():
    result = stgnt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stgnt_alias():
    from moirais.fn.stgnt import stgnt
    assert stgnt is stgnt
