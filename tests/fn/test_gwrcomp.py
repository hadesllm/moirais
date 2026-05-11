"""Tests for morie.fn.gwrcomp."""
import numpy as np
import pytest
from morie.fn.gwrcomp import gwrcomp


class TestGwrcomp:
    def test_basic(self):
        aicc_fixed=120.0; aicc_adapt=118.0
        result = gwrcomp(aicc_fixed, aicc_adapt)
        assert result is not None

    def test_returns_spatial_result(self):
        aicc_fixed=120.0; aicc_adapt=118.0
        result = gwrcomp(aicc_fixed, aicc_adapt)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        aicc_fixed=120.0; aicc_adapt=118.0
        result = gwrcomp(aicc_fixed, aicc_adapt)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
