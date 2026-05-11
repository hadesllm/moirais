"""Tests for rnflwdr."""
import numpy as np
import pytest
from morie.fn.rnflwdr import rnflwdr


def test_rnflwdr_basic():
    result = rnflwdr()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-FlowDir"


def test_rnflwdr_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnflwdr(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnflwdr_no_data():
    result = rnflwdr(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnflwdr_alias():
    from morie.fn.rnflwdr import rnflwdr
    assert rnflwdr is rnflwdr
