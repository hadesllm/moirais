"""Tests for morie.fn.siurt — SIU case rate."""

import pytest
from morie.fn.siurt import siu_case_rate
from morie.fn._containers import CrimeResult


class TestSiuCaseRate:
    def test_basic(self):
        r = siu_case_rate(25, 5000)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(5.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            siu_case_rate(10, 0)
