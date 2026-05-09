"""Tests for moirais.fn.sarspil."""
import numpy as np
import pytest
from moirais.fn.sarspil import sarspil


class TestSarspil:
    def test_basic(self):
        coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1
        result = sarspil(coef, rho, W)
        assert result is not None

    def test_returns_spatial_result(self):
        coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1
        result = sarspil(coef, rho, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        coef=np.array([1.0,0.5]); rho=0.2; W=np.eye(8)*0.1
        result = sarspil(coef, rho, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
