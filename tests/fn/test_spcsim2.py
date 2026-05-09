"""Tests for spcsim2."""
import numpy as np
import pytest
from moirais.fn.spcsim2 import spcsim2


def test_spcsim2_basic():
    result = spcsim2()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SpectralSim-2D"


def test_spcsim2_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = spcsim2(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_spcsim2_no_data():
    result = spcsim2(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_spcsim2_alias():
    from moirais.fn.spcsim2 import spcsim2
    assert spcsim2 is spcsim2
