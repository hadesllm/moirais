"""Tests for morie.fn.sdemlrt."""
import numpy as np
import pytest
from morie.fn.sdemlrt import sdemlrt


class TestSdemlrt:
    def test_basic(self):
        ll_sdem=-43.0; ll_sem=-50.0; df=2
        result = sdemlrt(ll_sdem, ll_sem, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_sdem=-43.0; ll_sem=-50.0; df=2
        result = sdemlrt(ll_sdem, ll_sem, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_sdem=-43.0; ll_sem=-50.0; df=2
        result = sdemlrt(ll_sdem, ll_sem, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
