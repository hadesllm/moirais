"""Tests for moirais.fn.mgwrdg."""
import numpy as np
import pytest
from moirais.fn.mgwrdg import mgwrdg


class TestMgwrdg:
    def test_basic(self):
        ll=-50.0; tr_S=5.0; n=30; k=3
        result = mgwrdg(ll, tr_S, n, k)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-50.0; tr_S=5.0; n=30; k=3
        result = mgwrdg(ll, tr_S, n, k)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-50.0; tr_S=5.0; n=30; k=3
        result = mgwrdg(ll, tr_S, n, k)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
