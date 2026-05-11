"""Tests for nnwgt."""
import numpy as np
import pytest
from morie.fn.nnwgt import nnwgt


def test_nnwgt_basic():
    result = nnwgt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "NatNeighbor-AreaWeight"


def test_nnwgt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = nnwgt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_nnwgt_no_data():
    result = nnwgt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_nnwgt_alias():
    from morie.fn.nnwgt import nnwgt
    assert nnwgt is nnwgt
