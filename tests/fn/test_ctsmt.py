"""Tests for ctsmt."""
import numpy as np
import pytest
from moirais.fn.ctsmt import ctsmt


def test_ctsmt_basic():
    result = ctsmt()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "Contour-Smooth"


def test_ctsmt_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = ctsmt(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_ctsmt_no_data():
    result = ctsmt(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_ctsmt_alias():
    from moirais.fn.ctsmt import ctsmt
    assert ctsmt is ctsmt
