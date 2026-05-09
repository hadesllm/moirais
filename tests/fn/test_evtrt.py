"""Tests for moirais.fn.evtrt — event rate."""
import numpy as np
import pytest

from moirais.fn.evtrt import event_rate, evtrt


def test_overall_rate():
    events = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = event_rate(events, duration=1.0, window=0.5)
    assert result.value == 5.0


def test_empty_events():
    events = np.array([])
    result = event_rate(events, duration=1.0, window=0.5)
    assert result.value == 0.0


def test_binned_rates():
    events = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    result = event_rate(events, duration=1.0, window=0.5)
    assert len(result.extra["rates"]) == 2


def test_alias():
    assert evtrt is event_rate
