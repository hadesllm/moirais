"""Tests for moirais.fn.r0 -- basic reproduction number."""

import pytest
from moirais.fn.r0 import basic_reproduction_number


class TestR0:
    def test_direct(self):
        """beta=0.3, gamma=0.1 => R0=3.0."""
        res = basic_reproduction_number(beta=0.3, gamma=0.1)
        assert res.measure == "R0"
        assert res.estimate == pytest.approx(3.0)

    def test_from_attack_rate(self):
        """Known: R0=2 gives attack rate ~0.7968."""
        res = basic_reproduction_number(attack_rate=0.7968)
        assert res.estimate == pytest.approx(2.0, abs=0.05)

    def test_invalid_raises(self):
        """No args should raise."""
        with pytest.raises(ValueError):
            basic_reproduction_number()
