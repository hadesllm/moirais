"""Tests for moirais.fn.slxwald."""
import numpy as np
import pytest
from moirais.fn.slxwald import slxwald


class TestSlxwald:
    def test_basic(self):
        theta=np.array([0.1,0.2]); se_theta=np.array([0.05,0.06])
        result = slxwald(theta, se_theta)
        assert result is not None

    def test_returns_spatial_result(self):
        theta=np.array([0.1,0.2]); se_theta=np.array([0.05,0.06])
        result = slxwald(theta, se_theta)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        theta=np.array([0.1,0.2]); se_theta=np.array([0.05,0.06])
        result = slxwald(theta, se_theta)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
