"""Tests for moirais.fn.sdmlrt."""
import numpy as np
import pytest
from moirais.fn.sdmlrt import sdmlrt


class TestSdmlrt:
    def test_basic(self):
        ll_sdm=-42.0; ll_sar=-50.0; df=2
        result = sdmlrt(ll_sdm, ll_sar, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_sdm=-42.0; ll_sar=-50.0; df=2
        result = sdmlrt(ll_sdm, ll_sar, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_sdm=-42.0; ll_sar=-50.0; df=2
        result = sdmlrt(ll_sdm, ll_sar, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
