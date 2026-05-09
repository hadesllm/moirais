"""Tests for moirais.fn.sppvar."""
import numpy as np
import pytest
from moirais.fn.sppvar import sppvar


class TestSppvar:
    def test_basic(self):
        np.random.seed(138); n=15; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppvar(resid, unit_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(138); n=15; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppvar(resid, unit_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(138); n=15; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T)
        result = sppvar(resid, unit_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
