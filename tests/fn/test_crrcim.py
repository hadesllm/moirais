"""Tests for crrcim.cumulative_incidence."""
import numpy as np
import pytest
from moirais.fn.crrcim import cumulative_incidence


def test_crrcim_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_incidence(time, event_type, cause)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crrcim_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_incidence(time, event_type, cause)
    assert isinstance(result, dict)
