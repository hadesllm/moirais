"""Tests for moirais.fn.sppbic."""
import numpy as np
import pytest
from moirais.fn.sppbic import sppbic


class TestSppbic:
    def test_basic(self):
        ll=-80.0; k=6; n=60
        result = sppbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-80.0; k=6; n=60
        result = sppbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-80.0; k=6; n=60
        result = sppbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
