"""Tests for morie.fn.crtcv — civil liberties."""

import pytest
from morie.fn.crtcv import court_civil_liberties
from morie.fn._containers import CrimeResult


class TestCivilLiberties:
    def test_basic(self):
        r = court_civil_liberties(100, 30)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            court_civil_liberties(0, 0)

    def test_granted_exceeds(self):
        with pytest.raises(ValueError):
            court_civil_liberties(10, 20)
