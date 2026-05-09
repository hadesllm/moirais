"""Tests for moirais.fn.prgcm — program completion."""

import pytest
from moirais.fn.prgcm import program_completion
from moirais.fn._containers import CrimeResult


class TestProgramCompletion:
    def test_basic(self):
        r = program_completion(70, 100)
        assert r.rate == pytest.approx(0.7)
        assert r.extra["dropout_rate"] == pytest.approx(0.3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            program_completion(5, 0)
