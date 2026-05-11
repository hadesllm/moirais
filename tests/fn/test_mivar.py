"""Tests for morie.fn.mivar."""
import numpy as np
import pytest
from morie.fn.mivar import mivar


class TestMivar:
    def test_basic(self):
        n=20; S0=10.0; S1=20.0; S2=40.0
        result = mivar(n, S0, S1, S2)
        assert result is not None

    def test_returns_spatial_result(self):
        n=20; S0=10.0; S1=20.0; S2=40.0
        result = mivar(n, S0, S1, S2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        n=20; S0=10.0; S1=20.0; S2=40.0
        result = mivar(n, S0, S1, S2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
