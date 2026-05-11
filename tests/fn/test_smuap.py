"""Tests for smuap.py - SMUAP point process EMG simulation."""
import numpy as np
from morie.fn.smuap import smuap_point_process_fn, smuap


def test_smuap_returns_signal_result():
    result = smuap_point_process_fn(n_mus=5, fs=1000.0, duration=0.5)
    assert result.name == "smuap_point_process"
    assert result.fs == 1000.0
    assert result.n_samples == 500


def test_smuap_emg_nonzero():
    result = smuap_point_process_fn(n_mus=10, duration=0.5)
    assert np.any(result.filtered != 0)


def test_smuap_events_in_extra():
    result = smuap_point_process_fn(n_mus=3, duration=0.2)
    assert "events" in result.extra
    assert len(result.extra["events"]) == result.n_samples


def test_smuap_alias():
    result = smuap(n_mus=2, duration=0.1)
    assert result.name == "smuap_point_process"
