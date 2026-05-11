"""Tests for morie.fn.sembic."""
import numpy as np
import pytest
from morie.fn.sembic import sembic


class TestSembic:
    def test_basic(self):
        ll=-60.0; k=4; n=50
        result = sembic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-60.0; k=4; n=50
        result = sembic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-60.0; k=4; n=50
        result = sembic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
