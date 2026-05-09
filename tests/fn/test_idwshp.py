"""Tests for idwshp."""
import numpy as np
import pytest
from moirais.fn.idwshp import idwshp


def test_idwshp_basic():
    result = idwshp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Shepard"


def test_idwshp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwshp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwshp_no_data():
    result = idwshp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwshp_alias():
    from moirais.fn.idwshp import idwshp
    assert idwshp is idwshp
