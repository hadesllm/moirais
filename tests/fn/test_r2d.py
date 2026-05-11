"""Tests for morie.fn.r2d -- Convert Pearson r to Cohen's d."""

import numpy as np
import pytest
from morie.fn.r2d import r_to_d


class TestRToD:
    def test_zero_r_gives_zero_d(self):
        """r=0 should give d=0."""
        assert r_to_d(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_positive_r_positive_d(self):
        """Positive r gives positive d."""
        assert r_to_d(0.5) > 0

    def test_roundtrip_with_d2r(self):
        """r -> d -> r should recover original r."""
        from morie.fn.d2r import d_to_r
        r_orig = 0.4
        r_back = d_to_r(r_to_d(r_orig))
        assert r_back == pytest.approx(r_orig, abs=1e-6)
