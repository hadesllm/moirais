"""Tests for moirais.fn.sdemjac."""
import numpy as np
import pytest
from moirais.fn.sdemjac import sdemjac


class TestSdemjac:
    def test_basic(self):
        W=np.eye(10)*0.1; lam=0.3
        result = sdemjac(W, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.1; lam=0.3
        result = sdemjac(W, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.1; lam=0.3
        result = sdemjac(W, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
