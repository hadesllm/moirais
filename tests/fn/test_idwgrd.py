"""Tests for idwgrd."""
import numpy as np
import pytest
from moirais.fn.idwgrd import idwgrd


def test_idwgrd_basic():
    result = idwgrd()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Grid"


def test_idwgrd_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwgrd(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwgrd_no_data():
    result = idwgrd(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwgrd_alias():
    from moirais.fn.idwgrd import idwgrd
    assert idwgrd is idwgrd
