"""Tests for moirais.fn.slximp."""
import numpy as np
import pytest
from moirais.fn.slximp import slximp


class TestSlximp:
    def test_basic(self):
        coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2])
        result = slximp(coef, theta)
        assert result is not None

    def test_returns_spatial_result(self):
        coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2])
        result = slximp(coef, theta)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        coef=np.array([0.5,0.3]); theta=np.array([0.1,0.2])
        result = slximp(coef, theta)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
