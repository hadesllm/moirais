"""Tests for morie.fn.caraic."""
import numpy as np
import pytest
from morie.fn.caraic import caraic


class TestCaraic:
    def test_basic(self):
        ll=-55.0; k=3; n=40
        result = caraic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-55.0; k=3; n=40
        result = caraic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-55.0; k=3; n=40
        result = caraic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
