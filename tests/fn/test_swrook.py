"""Tests for morie.fn.swrook."""
import numpy as np
import pytest
from morie.fn.swrook import swrook


class TestSwrook:
    def test_basic(self):
        nrow=4; ncol=4
        result = swrook(nrow, ncol)
        assert result is not None

    def test_returns_spatial_result(self):
        nrow=4; ncol=4
        result = swrook(nrow, ncol)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        nrow=4; ncol=4
        result = swrook(nrow, ncol)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
