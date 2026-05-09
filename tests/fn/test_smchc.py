"""Tests for moirais.fn.smchc -- simulate choice probabilities."""
import numpy as np
from moirais.fn.smchc import simulate_choice_prob, smchc


def test_alias():
    assert smchc is simulate_choice_prob


def test_smoke():
    u_yea = np.array([2.0, -1.0, 0.5])
    u_nay = np.array([0.0, 0.0, 0.0])
    r = simulate_choice_prob(u_yea, u_nay)
    assert r.name == "simulate_choice_prob"
    assert 0.0 < r.value < 1.0
    assert len(r.extra["prob_yea"]) == 3


def test_equal_utility():
    r = simulate_choice_prob([0, 0], [0, 0])
    assert abs(r.value - 0.5) < 1e-10
