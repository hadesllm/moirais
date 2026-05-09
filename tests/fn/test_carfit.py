"""Tests for moirais.fn.carfit."""
import numpy as np
import pytest
from moirais.fn.carfit import carfit


class TestCarfit:
    def test_basic(self):
        ll=-50.0; p_d=5.0
        result = carfit(ll, p_d)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-50.0; p_d=5.0
        result = carfit(ll, p_d)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-50.0; p_d=5.0
        result = carfit(ll, p_d)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
