"""Tests for morie.fn.scpaic."""
import numpy as np
import pytest
from morie.fn.scpaic import scpaic


class TestScpaic:
    def test_basic(self):
        ll=-70.0; k=5; n=40
        result = scpaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-70.0; k=5; n=40
        result = scpaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-70.0; k=5; n=40
        result = scpaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
