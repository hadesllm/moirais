"""Tests for morie.fn.phoen -- structural break detection."""

import numpy as np
from morie.fn.phoen import phoenix_break, phoen
from morie.fn._containers import TimeSeriesResult


class TestPhoen:
    def test_alias(self):
        assert phoen is phoenix_break

    def test_detects_break(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
        r = phoenix_break(y, min_segment=10)
        assert isinstance(r, TimeSeriesResult)
        assert len(r.extra["breakpoints"]) >= 1

    def test_no_break(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.1, 50)
        r = phoenix_break(y, min_segment=10, penalty=5.0)
        assert len(r.extra["breakpoints"]) == 0
