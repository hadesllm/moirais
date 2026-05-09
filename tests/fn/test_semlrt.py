"""Tests for moirais.fn.semlrt."""
import numpy as np
import pytest
from moirais.fn.semlrt import semlrt


class TestSemlrt:
    def test_basic(self):
        ll_sem=-44.0; ll_ols=-50.0; df=1
        result = semlrt(ll_sem, ll_ols, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_sem=-44.0; ll_ols=-50.0; df=1
        result = semlrt(ll_sem, ll_ols, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_sem=-44.0; ll_ols=-50.0; df=1
        result = semlrt(ll_sem, ll_ols, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
