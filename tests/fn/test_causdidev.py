"""Tests for causdidev.causal_did_eventstudy."""
import numpy as np
import pytest
from moirais.fn.causdidev import causal_did_eventstudy


def test_causdidev_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    K_event_time = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_eventstudy(Y_panel, K_event_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdidev_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    K_event_time = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_eventstudy(Y_panel, K_event_time)
    assert isinstance(result, dict)
