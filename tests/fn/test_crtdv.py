"""Tests for morie.fn.crtdv — diversion rate."""

import pytest
from morie.fn.crtdv import court_diversion
from morie.fn._containers import CrimeResult


class TestDiversion:
    def test_basic(self):
        r = court_diversion(40, 100)
        assert r.rate == pytest.approx(0.4)

    def test_invalid(self):
        with pytest.raises(ValueError):
            court_diversion(5, 0)
