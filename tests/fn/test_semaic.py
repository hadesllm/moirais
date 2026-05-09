"""Tests for moirais.fn.semaic."""
import numpy as np
import pytest
from moirais.fn.semaic import semaic


class TestSemaic:
    def test_basic(self):
        ll=-60.0; k=4; n=50
        result = semaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-60.0; k=4; n=50
        result = semaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-60.0; k=4; n=50
        result = semaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
