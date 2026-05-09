"""Tests for idwrad."""
import numpy as np
import pytest
from moirais.fn.idwrad import idwrad


def test_idwrad_basic():
    result = idwrad()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "IDW-RadialSearch"


def test_idwrad_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = idwrad(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_idwrad_no_data():
    result = idwrad(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_idwrad_alias():
    from moirais.fn.idwrad import idwrad
    assert idwrad is idwrad
