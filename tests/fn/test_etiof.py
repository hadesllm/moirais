"""Tests for moirais.fn.etiof — Etiologic fraction."""

import pytest

from moirais.fn.etiof import etiologic_fraction


class TestEtiologicFraction:
    def test_rr_2(self):
        res = etiologic_fraction(2.0)
        assert res.estimate == pytest.approx(0.5)

    def test_rr_1(self):
        res = etiologic_fraction(1.0)
        assert res.estimate == pytest.approx(0.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            etiologic_fraction(0)
