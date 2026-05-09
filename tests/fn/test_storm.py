"""Tests for moirais.fn.storm -- Markov chain weather model."""

import numpy as np
from moirais.fn.storm import markov_weather, storm
from moirais.fn._containers import DescriptiveResult


class TestStorm:
    def test_alias(self):
        assert storm is markov_weather

    def test_two_state(self):
        seq = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
        r = markov_weather(seq, seed=42)
        assert isinstance(r, DescriptiveResult)
        P = r.value["transition_matrix"]
        assert P.shape == (2, 2)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_forecast_length(self):
        seq = [0, 1, 2, 0, 1, 2, 0, 1]
        r = markov_weather(seq, forecast_steps=10, seed=0)
        assert len(r.value["forecast"]) == 10

    def test_stationary(self):
        seq = [0, 1] * 50
        r = markov_weather(seq)
        stat = r.value["stationary"]
        assert abs(stat.sum() - 1.0) < 1e-10
