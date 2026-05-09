"""Tests for moirais.fn.mtosp — speed analysis."""

import pytest
import numpy as np
from moirais.fn.mtosp import mto_speed_analysis
from moirais.fn._containers import DescriptiveResult


class TestSpeedAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = mto_speed_analysis(rng.normal(100, 15, 500))
        assert isinstance(r, DescriptiveResult)
        assert r.extra["mean"] == pytest.approx(100, abs=5)

    def test_over_limit(self):
        r = mto_speed_analysis([50, 60, 70, 80, 90], speed_limit=70)
        assert r.extra["pct_over_limit"] == pytest.approx(0.4)
