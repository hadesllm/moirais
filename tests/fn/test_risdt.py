"""Tests for morie.fn.risdt — rise time detection."""
import numpy as np
import pytest

from morie.fn.risdt import rise_time_detect, risdt


def test_ramp_rise():
    x = np.concatenate([np.zeros(10), np.linspace(0, 1, 20), np.ones(10)])
    events = [(10, 29)]
    result = rise_time_detect(x, events, threshold=0.1)
    assert result.value > 0


def test_flat_event():
    x = np.ones(50)
    events = [(10, 20)]
    result = rise_time_detect(x, events, threshold=0.1)
    assert result.value == 0.0


def test_multiple_events():
    x = np.concatenate([np.linspace(0, 1, 10), np.zeros(5), np.linspace(0, 2, 10)])
    events = [(0, 9), (15, 24)]
    result = rise_time_detect(x, events, threshold=0.1)
    assert result.extra["n_events"] == 2


def test_alias():
    assert risdt is rise_time_detect
