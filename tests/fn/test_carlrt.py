"""Tests for moirais.fn.carlrt."""
import numpy as np
import pytest
from moirais.fn.carlrt import carlrt


class TestCarlrt:
    def test_basic(self):
        ll_car=-40.0; ll_null=-50.0; df=1
        result = carlrt(ll_car, ll_null, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_car=-40.0; ll_null=-50.0; df=1
        result = carlrt(ll_car, ll_null, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_car=-40.0; ll_null=-50.0; df=1
        result = carlrt(ll_car, ll_null, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
