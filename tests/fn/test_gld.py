"""Tests for morie.fn.gld -- Glass's delta effect size."""

import pytest
from morie.fn.gld import glass_delta
from morie.fn._containers import ESRes


class TestGlassDelta:
    def test_known_groups(self):
        """Well-separated groups yield large delta."""
        x = [10.0, 11.0, 12.0, 13.0, 14.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = glass_delta(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate > 2.0

    def test_identical_groups_zero(self):
        """Identical groups give delta = 0."""
        x = [5.0, 6.0, 7.0, 8.0]
        result = glass_delta(x, x)
        assert result.estimate == pytest.approx(0.0, abs=1e-10)

    def test_control_choice_matters(self):
        """Changing control group changes denominator."""
        x = [10.0, 11.0, 12.0]
        y = [1.0, 1.5, 2.0]  # smaller SD
        res_y = glass_delta(x, y, control="y")
        res_x = glass_delta(x, y, control="x")
        assert res_y.estimate != pytest.approx(res_x.estimate, abs=0.1)
