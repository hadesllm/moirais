"""Tests for moirais.fn.tdehn — Dehn twist."""

import pytest

from moirais.fn.tdehn import dehn_twist


class TestDehnTwist:
    def test_a_twist(self):
        r = dehn_twist(curve_type="a", n=1)
        assert r.extra["matrix"] == [[1, 0], [1, 1]]

    def test_b_twist(self):
        r = dehn_twist(curve_type="b", n=1)
        assert r.extra["matrix"] == [[1, 1], [0, 1]]

    def test_zero_twist(self):
        r = dehn_twist(n=0)
        assert r.extra["classification"] == "identity"

    def test_invalid_curve(self):
        with pytest.raises(ValueError):
            dehn_twist(curve_type="c")
