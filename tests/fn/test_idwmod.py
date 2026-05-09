"""Tests for idwmod."""
import numpy as np
import pytest
from moirais.fn.idwmod import idwmod


def test_idwmod_basic():
    result = idwmod()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Modified"


def test_idwmod_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwmod(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwmod_no_data():
    result = idwmod(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwmod_alias():
    from moirais.fn.idwmod import idwmod
    assert idwmod is idwmod
