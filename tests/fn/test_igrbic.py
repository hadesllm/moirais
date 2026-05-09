"""Tests for moirais.fn.igrbic."""
import numpy as np
import pytest
from moirais.fn.igrbic import igrbic


class TestIgrbic:
    def test_basic(self):
        ll=-200.0; k=4; n=50
        result = igrbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-200.0; k=4; n=50
        result = igrbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-200.0; k=4; n=50
        result = igrbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
