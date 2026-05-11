"""Tests for morie.fn.prgef — program effect."""

import pytest
import numpy as np
from morie.fn.prgef import program_effect
from morie.fn._containers import ESRes


class TestProgramEffect:
    def test_positive_effect(self):
        rng = np.random.default_rng(42)
        pre = rng.normal(50, 10, 100)
        post = pre + rng.normal(5, 3, 100)
        r = program_effect(pre, post)
        assert isinstance(r, ESRes)
        assert r.estimate > 0

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            program_effect([1, 2], [1])
