"""Tests for moirais.fn.gnslrt."""
import numpy as np
import pytest
from moirais.fn.gnslrt import gnslrt


class TestGnslrt:
    def test_basic(self):
        ll_gns=-41.0; ll_sdm=-50.0; df=1
        result = gnslrt(ll_gns, ll_sdm, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_gns=-41.0; ll_sdm=-50.0; df=1
        result = gnslrt(ll_gns, ll_sdm, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_gns=-41.0; ll_sdm=-50.0; df=1
        result = gnslrt(ll_gns, ll_sdm, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
