"""Tests for morie.fn.mgwraic."""
import numpy as np
import pytest
from morie.fn.mgwraic import mgwraic


class TestMgwraic:
    def test_basic(self):
        ll=-55.0; tr_S=6.0; n=30
        result = mgwraic(ll, tr_S, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-55.0; tr_S=6.0; n=30
        result = mgwraic(ll, tr_S, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-55.0; tr_S=6.0; n=30
        result = mgwraic(ll, tr_S, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
