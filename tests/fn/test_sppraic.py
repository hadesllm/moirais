"""Tests for moirais.fn.sppraic."""
import numpy as np
import pytest
from moirais.fn.sppraic import sppraic


class TestSppraic:
    def test_basic(self):
        ll=-45.0; k=4; n=30
        result = sppraic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-45.0; k=4; n=30
        result = sppraic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-45.0; k=4; n=30
        result = sppraic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
