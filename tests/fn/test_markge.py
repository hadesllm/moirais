"""Tests for morie.fn.markge -- Markov chain generation."""

import numpy as np
from morie.fn.markge import markov_generate, markge
from morie.fn._containers import DescriptiveResult


class TestMarkge:
    def test_alias(self):
        assert markge is markov_generate

    def test_deterministic(self):
        P = np.array([[0, 1], [1, 0]], dtype=float)
        r = markov_generate(P, n_steps=6, start_state=0)
        assert isinstance(r, DescriptiveResult)
        expected = [0, 1, 0, 1, 0, 1]
        np.testing.assert_array_equal(r.value, expected)

    def test_stationary(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        r = markov_generate(P, n_steps=10000, start_state=0)
        emp = r.extra["empirical_stationary"]
        assert abs(emp[0] - 0.5) < 0.1
