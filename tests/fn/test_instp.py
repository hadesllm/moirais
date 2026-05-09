"""Tests for instp.py - Instantaneous phase."""
import numpy as np
from moirais.fn.instp import instantaneous_phase, instp


def test_instp_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = instantaneous_phase(x)
    assert result.name == "instantaneous_phase"
    assert "phase" in result.extra


def test_instp_monotonic_for_sine():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t)
    result = instantaneous_phase(x)
    phase = result.extra["phase"]
    diffs = np.diff(phase[32:224])
    assert np.all(diffs > 0)


def test_instp_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = instp(x)
    assert result.name == "instantaneous_phase"
