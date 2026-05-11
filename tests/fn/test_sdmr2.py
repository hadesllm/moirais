"""Tests for morie.fn.sdmr2."""
import numpy as np
import pytest
from morie.fn.sdmr2 import sdmr2


class TestSdmr2:
    def test_basic(self):
        ll_model=-42.0; ll_null=-60.0; n=50
        result = sdmr2(ll_model, ll_null, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_model=-42.0; ll_null=-60.0; n=50
        result = sdmr2(ll_model, ll_null, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_model=-42.0; ll_null=-60.0; n=50
        result = sdmr2(ll_model, ll_null, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
