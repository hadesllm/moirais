"""Tests for moirais.fn.sacaic."""
import numpy as np
import pytest
from moirais.fn.sacaic import sacaic


class TestSacaic:
    def test_basic(self):
        ll=-57.0; k=6; n=50
        result = sacaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-57.0; k=6; n=50
        result = sacaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-57.0; k=6; n=50
        result = sacaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
