"""Tests for hrvar -- HRV time-domain metrics."""
import numpy as np
import pytest
from morie.fn.hrvar import hrvar
from morie.fn._containers import DescriptiveResult


def test_hrvar_basic():
    rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.80, 0.78, 0.82])
    result = hrvar(rr)
    assert isinstance(result, DescriptiveResult)
    assert result.extra["sdnn"] > 0
    assert result.extra["rmssd"] > 0


def test_hrvar_pnn50():
    rr = np.array([0.8, 0.9, 0.8, 0.9, 0.8, 0.9])
    result = hrvar(rr)
    assert result.extra["pnn50"] == 1.0


def test_hrvar_too_short():
    with pytest.raises(ValueError):
        hrvar(np.array([0.8]))


def test_hrvar_mean_hr():
    rr = np.full(10, 1.0)
    result = hrvar(rr)
    assert abs(result.extra["mean_hr"] - 60.0) < 0.1
