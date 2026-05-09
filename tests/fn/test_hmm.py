"""Tests for hidden_markov."""
import numpy as np, pytest
from moirais.fn.hmm import hidden_markov

class TestHMM:
    def test_basic(self):
        rng = np.random.default_rng(0)
        obs = rng.integers(0, 3, 100)
        r = hidden_markov(obs, n_states=2, seed=0)
        assert r.name == "hmm"
        assert np.isfinite(r.value)

    def test_state_counts(self):
        obs = np.array([0, 1, 0, 1, 0, 1, 2, 2, 2, 0] * 10)
        r = hidden_markov(obs, n_states=2, seed=1)
        assert sum(r.extra["state_counts"]) == 100
