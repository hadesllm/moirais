"""Tests for morie.fn.sdemaic."""
import numpy as np
import pytest
from morie.fn.sdemaic import sdemaic


class TestSdemaic:
    def test_basic(self):
        ll=-56.0; k=6; n=50
        result = sdemaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-56.0; k=6; n=50
        result = sdemaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-56.0; k=6; n=50
        result = sdemaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
