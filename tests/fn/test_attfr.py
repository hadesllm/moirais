"""Tests for moirais.fn.attfr — Attributable fraction."""

import pytest

from moirais.fn.attfr import attributable_fraction


class TestAttributableFraction:
    def test_known_paf(self):
        res = attributable_fraction(2.0, 0.3)
        assert 0 < res.estimate < 1

    def test_no_effect(self):
        res = attributable_fraction(1.0, 0.5)
        assert res.estimate == pytest.approx(0.0)

    def test_invalid_rr(self):
        with pytest.raises(ValueError):
            attributable_fraction(-1.0, 0.3)
