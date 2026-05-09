"""Tests for moirais.fn.sdmaic."""
import numpy as np
import pytest
from moirais.fn.sdmaic import sdmaic


class TestSdmaic:
    def test_basic(self):
        ll=-58.0; k=6; n=50
        result = sdmaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-58.0; k=6; n=50
        result = sdmaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-58.0; k=6; n=50
        result = sdmaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
