"""Tests for moirais.fn.prgen — program enrollment."""

import pytest
from moirais.fn.prgen import program_enrollment
from moirais.fn._containers import CrimeResult


class TestProgramEnrollment:
    def test_basic(self):
        r = program_enrollment(80, 100)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.8)

    def test_invalid(self):
        with pytest.raises(ValueError):
            program_enrollment(10, 0)
