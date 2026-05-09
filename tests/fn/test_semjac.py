"""Tests for moirais.fn.semjac."""
import numpy as np
import pytest
from moirais.fn.semjac import semjac


class TestSemjac:
    def test_basic(self):
        W=np.eye(10)*0.1; lam=0.3
        result = semjac(W, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; lam=0.3
        result = semjac(W, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; lam=0.3
        result = semjac(W, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
