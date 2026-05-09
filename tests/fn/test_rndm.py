"""Tests for moirais.fn.rndm — Block randomization."""

import numpy as np
import pytest

from moirais.fn.rndm import randomization


class TestRandomization:
    def test_correct_length(self):
        res = randomization(100, seed=42)
        assert len(res.extra["sequence"]) == 100

    def test_balanced_arms(self):
        res = randomization(1000, seed=42)
        seq = np.array(res.extra["sequence"])
        assert abs(np.mean(seq == 0) - 0.5) < 0.05

    def test_three_arms(self):
        res = randomization(99, n_arms=3, block_sizes=[6], seed=42)
        seq = np.array(res.extra["sequence"])
        assert set(np.unique(seq)) == {0, 1, 2}
