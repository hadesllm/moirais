"""Test event_align (evtag)."""
import numpy as np
from moirais.fn.evtag import event_align, evtag
from moirais.fn._containers import DescriptiveResult


class TestEventAlign:
    def test_basic(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        events = np.array([100, 200, 300])
        result = event_align(signal, events, window=20)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "event_align"

    def test_empty_events(self):
        signal = np.ones(100)
        result = event_align(signal, np.array([]))
        assert result.value == 0.0

    def test_shifts_computed(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        events = np.array([100, 250, 400])
        result = event_align(signal, events, window=30)
        assert len(result.extra["shifts"]) == 3

    def test_alias(self):
        assert evtag is event_align
