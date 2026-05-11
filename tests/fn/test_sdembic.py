"""Tests for morie.fn.sdembic."""
import numpy as np
import pytest
from morie.fn.sdembic import sdembic


class TestSdembic:
    def test_basic(self):
        ll=-56.0; k=6; n=50
        result = sdembic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-56.0; k=6; n=50
        result = sdembic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-56.0; k=6; n=50
        result = sdembic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
