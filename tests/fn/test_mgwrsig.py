"""Tests for morie.fn.mgwrsig."""
import numpy as np
import pytest
from morie.fn.mgwrsig import mgwrsig


class TestMgwrsig:
    def test_basic(self):
        np.random.seed(124); resid=np.random.randn(15); tr_S=5.0; n=15
        result = mgwrsig(resid, tr_S, n)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(124); resid=np.random.randn(15); tr_S=5.0; n=15
        result = mgwrsig(resid, tr_S, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(124); resid=np.random.randn(15); tr_S=5.0; n=15
        result = mgwrsig(resid, tr_S, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
