"""Tests for morie.fn.d2r -- Convert Cohen's d to Pearson r."""

import pytest
from morie.fn.d2r import d_to_r


class TestDToR:
    def test_zero_d_gives_zero_r(self):
        """d=0 should give r=0."""
        assert d_to_r(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_positive_d_positive_r(self):
        """Positive d gives positive r."""
        assert d_to_r(0.8) > 0

    def test_with_sample_sizes(self):
        """Providing n1, n2 changes the correction factor a."""
        r_default = d_to_r(0.5)
        r_unequal = d_to_r(0.5, n1=20, n2=80)
        assert r_default != pytest.approx(r_unequal, abs=0.001)
