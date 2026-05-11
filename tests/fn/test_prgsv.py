"""Tests for morie.fn.prgsv — program survival."""

import pytest
import numpy as np
from morie.fn.prgsv import program_survival
from morie.fn._containers import DescriptiveResult


class TestProgramSurvival:
    def test_basic(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(12, 100)
        completed = rng.binomial(1, 0.7, 100)
        r = program_survival(times, completed)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n"] == 100

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            program_survival([], [])
