"""Tests for moirais.fn.herd -- herd immunity threshold."""

import pytest
from moirais.fn.herd import herd_immunity_threshold


class TestHerdImmunity:
    def test_r0_3(self):
        """R0=3 => HIT = 1 - 1/3 = 0.667."""
        res = herd_immunity_threshold(R0=3.0)
        assert res.estimate == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_r0_1(self):
        """R0=1 => HIT = 0 (no epidemic spread)."""
        res = herd_immunity_threshold(R0=1.0)
        assert res.estimate == pytest.approx(0.0)

    def test_r0_negative_raises(self):
        """R0 <= 0 should raise."""
        with pytest.raises(ValueError):
            herd_immunity_threshold(R0=-1.0)
