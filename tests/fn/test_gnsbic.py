"""Tests for moirais.fn.gnsbic."""
import numpy as np
import pytest
from moirais.fn.gnsbic import gnsbic


class TestGnsbic:
    def test_basic(self):
        ll=-55.0; k=7; n=50
        result = gnsbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-55.0; k=7; n=50
        result = gnsbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-55.0; k=7; n=50
        result = gnsbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
