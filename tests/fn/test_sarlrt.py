"""Tests for moirais.fn.sarlrt."""
import numpy as np
import pytest
from moirais.fn.sarlrt import sarlrt


class TestSarlrt:
    def test_basic(self):
        ll_sar=-45.0; ll_ols=-50.0; df=1
        result = sarlrt(ll_sar, ll_ols, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_sar=-45.0; ll_ols=-50.0; df=1
        result = sarlrt(ll_sar, ll_ols, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_sar=-45.0; ll_ols=-50.0; df=1
        result = sarlrt(ll_sar, ll_ols, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
