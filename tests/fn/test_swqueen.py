"""Tests for morie.fn.swqueen."""
import numpy as np
import pytest
from morie.fn.swqueen import swqueen


class TestSwqueen:
    def test_basic(self):
        nrow=4; ncol=4
        result = swqueen(nrow, ncol)
        assert result is not None

    def test_returns_spatial_result(self):
        nrow=4; ncol=4
        result = swqueen(nrow, ncol)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        nrow=4; ncol=4
        result = swqueen(nrow, ncol)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
