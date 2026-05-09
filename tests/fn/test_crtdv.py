"""Tests for moirais.fn.crtdv — diversion rate."""

import pytest
from moirais.fn.crtdv import court_diversion
from moirais.fn._containers import CrimeResult


class TestDiversion:
    def test_basic(self):
        r = court_diversion(40, 100)
        assert r.rate == pytest.approx(0.4)

    def test_invalid(self):
        with pytest.raises(ValueError):
            court_diversion(5, 0)
