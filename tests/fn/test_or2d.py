"""Tests for moirais.fn.or2d -- Convert odds ratio to Cohen's d."""

import pytest
from moirais.fn.or2d import or_to_d


class TestORToD:
    def test_or_one_gives_zero(self):
        """OR=1 (no effect) should give d=0."""
        assert or_to_d(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_or_greater_positive_d(self):
        """OR > 1 gives positive d."""
        assert or_to_d(2.0) > 0

    def test_roundtrip_with_d2or(self):
        """d -> OR -> d should recover original d."""
        from moirais.fn.d2or import d_to_or
        d_orig = 0.6
        d_back = or_to_d(d_to_or(d_orig))
        assert d_back == pytest.approx(d_orig, abs=1e-10)
