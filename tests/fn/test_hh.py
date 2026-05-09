"""Tests for hh.py - Hodgkin-Huxley neuron model."""
import numpy as np
from moirais.fn.hh import hodgkin_huxley_fn, hh


def test_hh_returns_signal_result():
    result = hodgkin_huxley_fn(duration=10.0, dt=0.01)
    assert result.name == "hodgkin_huxley"
    assert result.n_samples == 1000
    assert result.fs == 100.0


def test_hh_produces_spikes():
    result = hodgkin_huxley_fn(duration=50.0, dt=0.01, I_ext=10.0)
    assert np.max(result.filtered) > 0


def test_hh_time_in_extra():
    result = hodgkin_huxley_fn(duration=5.0, dt=0.05)
    assert "time" in result.extra
    assert len(result.extra["time"]) == result.n_samples


def test_hh_alias():
    result = hh(duration=5.0, dt=0.05, I_ext=5.0)
    assert result.name == "hodgkin_huxley"
