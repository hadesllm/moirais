"""Tests for moirais.fn.mtobac — BAC analysis."""

import pytest
import numpy as np
from moirais.fn.mtobac import mto_bac_analysis
from moirais.fn._containers import DescriptiveResult


class TestBacAnalysis:
    def test_basic(self):
        r = mto_bac_analysis([0.05, 0.08, 0.10, 0.15, 0.20])
        assert isinstance(r, DescriptiveResult)
        assert r.extra["pct_over_limit"] == pytest.approx(0.6)

    def test_all_under(self):
        r = mto_bac_analysis([0.01, 0.02, 0.03])
        assert r.extra["pct_over_limit"] == pytest.approx(0.0)
