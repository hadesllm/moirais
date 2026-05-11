"""Tests for morie.fn.swisle."""
import numpy as np
import pytest
from morie.fn.swisle import swisle


class TestSwisle:
    def test_basic(self):
        W=np.block([[np.ones((3,3)),np.zeros((3,3))],[np.zeros((3,3)),np.ones((3,3))]]); np.fill_diagonal(W,0)
        result = swisle(W)
        assert result is not None

    def test_returns_spatial_result(self):
        W=np.block([[np.ones((3,3)),np.zeros((3,3))],[np.zeros((3,3)),np.ones((3,3))]]); np.fill_diagonal(W,0)
        result = swisle(W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        W=np.block([[np.ones((3,3)),np.zeros((3,3))],[np.zeros((3,3)),np.ones((3,3))]]); np.fill_diagonal(W,0)
        result = swisle(W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
