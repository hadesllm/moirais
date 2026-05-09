"""Tests for rfani."""
import numpy as np
import pytest
from moirais.fn.rfani import rfani


def test_rfani_basic():
    result = rfani()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "RF-Anisotropic"


def test_rfani_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = rfani(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_rfani_no_data():
    result = rfani(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_rfani_alias():
    from moirais.fn.rfani import rfani
    assert rfani is rfani
