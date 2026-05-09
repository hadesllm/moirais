"""Tests for idwmsh."""
import numpy as np
import pytest
from moirais.fn.idwmsh import idwmsh


def test_idwmsh_basic():
    result = idwmsh()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-ModifiedShepard"


def test_idwmsh_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwmsh(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwmsh_no_data():
    result = idwmsh(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwmsh_alias():
    from moirais.fn.idwmsh import idwmsh
    assert idwmsh is idwmsh
