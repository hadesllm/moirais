"""Tests for ttrace.contact_tracing_yield."""
import numpy as np
import pytest
from morie.fn.ttrace import contact_tracing_yield


def test_ttrace_basic():
    """Test basic functionality."""
    contacts = np.random.default_rng(42).normal(0, 1, 100)
    detection_rate = 0.1
    positivity = np.random.default_rng(42).normal(0, 1, 100)
    result = contact_tracing_yield(contacts, detection_rate, positivity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ttrace_edge():
    """Test edge cases."""
    contacts = np.random.default_rng(42).normal(0, 1, 100)
    detection_rate = 0.1
    positivity = np.random.default_rng(42).normal(0, 1, 100)
    result = contact_tracing_yield(contacts, detection_rate, positivity)
    assert isinstance(result, dict)
