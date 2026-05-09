"""Tests for moirais.fn.sppde."""
import numpy as np
import pytest
from moirais.fn.sppde import sppde


class TestSppde:
    def test_basic(self):
        np.random.seed(134); n=15; T=3; y=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppde(y, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(134); n=15; T=3; y=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppde(y, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(134); n=15; T=3; y=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppde(y, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
