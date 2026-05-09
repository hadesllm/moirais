"""Tests for moirais.fn.etagr -- Etiologic fraction."""

import pytest
from moirais.fn.etagr import etiologic_fraction


class TestEtiologicFraction:
    def test_known(self):
        res = etiologic_fraction(rr=2.0)
        assert res.estimate == pytest.approx(0.5)

    def test_rr_one(self):
        res = etiologic_fraction(rr=1.0)
        assert res.estimate == pytest.approx(0.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            etiologic_fraction(rr=-1)
