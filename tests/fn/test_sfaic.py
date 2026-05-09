"""Tests for moirais.fn.sfaic."""
import numpy as np
import pytest
from moirais.fn.sfaic import sfaic


class TestSfaic:
    def test_basic(self):
        ll=-50.0; k=5; n=30
        result = sfaic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-50.0; k=5; n=30
        result = sfaic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-50.0; k=5; n=30
        result = sfaic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
