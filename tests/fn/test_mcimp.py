"""Tests for mcimp."""
import numpy as np
import pytest
from moirais.fn.mcimp import mcimp


def test_mcimp_basic():
    result = mcimp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "MC-Importance"


def test_mcimp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = mcimp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_mcimp_no_data():
    result = mcimp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_mcimp_alias():
    from moirais.fn.mcimp import mcimp
    assert mcimp is mcimp
