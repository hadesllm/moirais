"""Tests for morie.fn.spprprd."""
import numpy as np
import pytest
from morie.fn.spprprd import spprprd


class TestSpprprd:
    def test_basic(self):
        np.random.seed(150); coef=np.array([0.5,0.3]); n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; rho=0.2
        result = spprprd(coef, X, W, rho)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(150); coef=np.array([0.5,0.3]); n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; rho=0.2
        result = spprprd(coef, X, W, rho)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(150); coef=np.array([0.5,0.3]); n=10; X=np.column_stack([np.ones(n),np.random.randn(n)]); W=np.eye(n)*0.2; rho=0.2
        result = spprprd(coef, X, W, rho)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
