"""Tests for morie.fn.faldt — fall time detection."""
import numpy as np
import pytest

from morie.fn.faldt import fall_time_detect, faldt


def test_ramp_down():
    x = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
    events = [(0, 19)]
    result = fall_time_detect(x, events, threshold=0.1)
    assert result.value > 0


def test_flat_event():
    x = np.ones(50)
    events = [(10, 20)]
    result = fall_time_detect(x, events, threshold=0.1)
    assert result.value == 0.0


def test_multiple_events():
    x = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10), np.zeros(5), np.linspace(0, 2, 10), np.linspace(2, 0, 10)])
    events = [(0, 19), (25, 44)]
    result = fall_time_detect(x, events, threshold=0.1)
    assert result.extra["n_events"] == 2


def test_alias():
    assert faldt is fall_time_detect
