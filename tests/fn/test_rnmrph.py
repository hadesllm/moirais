"""Tests for rnmrph."""
import numpy as np
import pytest
from morie.fn.rnmrph import rnmrph


def test_rnmrph_basic():
    result = rnmrph()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Raster-Morphological"


def test_rnmrph_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rnmrph(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rnmrph_no_data():
    result = rnmrph(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rnmrph_alias():
    from morie.fn.rnmrph import rnmrph
    assert rnmrph is rnmrph
