"""Tests for morie.fn.igraic."""
import numpy as np
import pytest
from morie.fn.igraic import igraic


class TestIgraic:
    def test_basic(self):
        ll=-200.0; k=4; n=50
        result = igraic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-200.0; k=4; n=50
        result = igraic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-200.0; k=4; n=50
        result = igraic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
