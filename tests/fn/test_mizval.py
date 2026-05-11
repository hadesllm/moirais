"""Tests for morie.fn.mizval."""
import numpy as np
import pytest
from morie.fn.mizval import mizval


class TestMizval:
    def test_basic(self):
        I=0.4; E_I=-0.05; Var_I=0.02
        result = mizval(I, E_I, Var_I)
        assert result is not None

    def test_returns_spatial_result(self):
        I=0.4; E_I=-0.05; Var_I=0.02
        result = mizval(I, E_I, Var_I)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        I=0.4; E_I=-0.05; Var_I=0.02
        result = mizval(I, E_I, Var_I)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
