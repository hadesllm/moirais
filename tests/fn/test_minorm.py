"""Tests for moirais.fn.minorm."""
import numpy as np
import pytest
from moirais.fn.minorm import minorm


class TestMinorm:
    def test_basic(self):
        I=0.3; n=20; S0=10.0; S1=20.0; S2=40.0
        result = minorm(I, n, S0, S1, S2)
        assert result is not None

    def test_returns_spatial_result(self):
        I=0.3; n=20; S0=10.0; S1=20.0; S2=40.0
        result = minorm(I, n, S0, S1, S2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        I=0.3; n=20; S0=10.0; S1=20.0; S2=40.0
        result = minorm(I, n, S0, S1, S2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
