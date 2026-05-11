"""Tests for morie.fn.sacbic."""
import numpy as np
import pytest
from morie.fn.sacbic import sacbic


class TestSacbic:
    def test_basic(self):
        ll=-57.0; k=6; n=50
        result = sacbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-57.0; k=6; n=50
        result = sacbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-57.0; k=6; n=50
        result = sacbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
