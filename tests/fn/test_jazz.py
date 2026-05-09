"""Tests for moirais.fn.jazz -- Markov chain generation."""

import numpy as np
from moirais.fn.jazz import markov_generate, jazz
from moirais.fn._containers import DescriptiveResult


class TestJazz:
    def test_alias(self):
        assert jazz is markov_generate

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
