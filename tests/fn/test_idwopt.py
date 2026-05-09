"""Tests for idwopt."""
import numpy as np
import pytest
from moirais.fn.idwopt import idwopt


def test_idwopt_basic():
    result = idwopt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Optimal"


def test_idwopt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwopt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwopt_no_data():
    result = idwopt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwopt_alias():
    from moirais.fn.idwopt import idwopt
    assert idwopt is idwopt
