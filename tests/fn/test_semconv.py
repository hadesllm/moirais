"""Tests for moirais.fn.semconv."""
import numpy as np
import pytest
from moirais.fn.semconv import semconv


class TestSemconv:
    def test_basic(self):
        W=np.eye(10)*0.2; lam=0.3
        result = semconv(W, lam)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.eye(10)*0.2; lam=0.3
        result = semconv(W, lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.eye(10)*0.2; lam=0.3
        result = semconv(W, lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
