"""Tests for moirais.fn.evtdn — event density."""
import numpy as np
import pytest

from moirais.fn.evtdn import event_density, evtdn


def test_single_event():
    events = np.array([0.5])
    result = event_density(events, duration=1.0, bandwidth=0.1)
    assert result.value > 0


def test_empty_events():
    events = np.array([])
    result = event_density(events, duration=1.0, bandwidth=0.1)
    assert result.value == 0.0


def test_density_shape():
    rng = np.random.default_rng(42)
    events = rng.uniform(0, 10, 20)
    result = event_density(events, duration=10.0, bandwidth=0.5)
    assert len(result.extra["t"]) == len(result.extra["density"])


def test_alias():
    assert evtdn is event_density
