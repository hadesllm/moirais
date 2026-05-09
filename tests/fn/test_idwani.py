"""Tests for idwani."""
import numpy as np
import pytest
from moirais.fn.idwani import idwani


def test_idwani_basic():
    result = idwani()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-Anisotropic"


def test_idwani_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwani(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwani_no_data():
    result = idwani(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwani_alias():
    from moirais.fn.idwani import idwani
    assert idwani is idwani
