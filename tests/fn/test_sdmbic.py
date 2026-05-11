"""Tests for morie.fn.sdmbic."""
import numpy as np
import pytest
from morie.fn.sdmbic import sdmbic


class TestSdmbic:
    def test_basic(self):
        ll=-58.0; k=6; n=50
        result = sdmbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-58.0; k=6; n=50
        result = sdmbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-58.0; k=6; n=50
        result = sdmbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
