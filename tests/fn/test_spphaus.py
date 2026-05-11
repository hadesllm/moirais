"""Tests for morie.fn.spphaus."""
import numpy as np
import pytest
from morie.fn.spphaus import spphaus


class TestSpphaus:
    def test_basic(self):
        np.random.seed(129); coef_fe=np.random.randn(2); coef_re=np.random.randn(2); vcov_fe=np.eye(2)*0.01; vcov_re=np.eye(2)*0.02
        result = spphaus(coef_fe, coef_re, vcov_fe, vcov_re)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(129); coef_fe=np.random.randn(2); coef_re=np.random.randn(2); vcov_fe=np.eye(2)*0.01; vcov_re=np.eye(2)*0.02
        result = spphaus(coef_fe, coef_re, vcov_fe, vcov_re)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(129); coef_fe=np.random.randn(2); coef_re=np.random.randn(2); vcov_fe=np.eye(2)*0.01; vcov_re=np.eye(2)*0.02
        result = spphaus(coef_fe, coef_re, vcov_fe, vcov_re)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
