"""Tests for morie.fn.saclrt."""
import numpy as np
import pytest
from morie.fn.saclrt import saclrt


class TestSaclrt:
    def test_basic(self):
        ll_sac=-42.0; ll_sar=-50.0; df=1
        result = saclrt(ll_sac, ll_sar, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_sac=-42.0; ll_sar=-50.0; df=1
        result = saclrt(ll_sac, ll_sar, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_sac=-42.0; ll_sar=-50.0; df=1
        result = saclrt(ll_sac, ll_sar, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
