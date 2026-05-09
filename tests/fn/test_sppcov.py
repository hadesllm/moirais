"""Tests for moirais.fn.sppcov."""
import numpy as np
import pytest
from moirais.fn.sppcov import sppcov


class TestSppcov:
    def test_basic(self):
        np.random.seed(143); n=12; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T); time_id=np.repeat(np.arange(T),n)
        result = sppcov(resid, unit_id, time_id)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(143); n=12; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T); time_id=np.repeat(np.arange(T),n)
        result = sppcov(resid, unit_id, time_id)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(143); n=12; T=3; resid=np.random.randn(n*T); unit_id=np.tile(np.arange(n),T); time_id=np.repeat(np.arange(T),n)
        result = sppcov(resid, unit_id, time_id)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
