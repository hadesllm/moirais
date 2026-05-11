"""Tests for morie.fn.sdemwld."""
import numpy as np
import pytest
from morie.fn.sdemwld import sdemwld


class TestSdemwld:
    def test_basic(self):
        lam=0.4; se_lam=0.1
        result = sdemwld(lam, se_lam)
        assert result is not None

    def test_returns_spatial_result(self):
        lam=0.4; se_lam=0.1
        result = sdemwld(lam, se_lam)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        lam=0.4; se_lam=0.1
        result = sdemwld(lam, se_lam)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
