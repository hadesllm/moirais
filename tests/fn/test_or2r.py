"""Tests for morie.fn.or2r -- Convert odds ratio to Pearson r."""

import pytest
from morie.fn.or2r import or_to_r


class TestORToR:
    def test_or_one_gives_zero(self):
        """OR=1 should give r near 0."""
        assert or_to_r(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_or_greater_positive_r(self):
        """OR > 1 gives positive r."""
        assert or_to_r(3.0) > 0

    def test_bounded(self):
        """r should be in (-1, 1) for finite OR."""
        r = or_to_r(5.0)
        assert -1.0 < r < 1.0
