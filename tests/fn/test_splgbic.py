"""Tests for morie.fn.splgbic."""
import numpy as np
import pytest
from morie.fn.splgbic import splgbic


class TestSplgbic:
    def test_basic(self):
        ll=-44.0; k=4; n=30
        result = splgbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-44.0; k=4; n=30
        result = splgbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-44.0; k=4; n=30
        result = splgbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
