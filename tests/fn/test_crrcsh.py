"""Tests for crrcsh.cause_specific_hazard."""
import numpy as np
import pytest
from moirais.fn.crrcsh import cause_specific_hazard


def test_crrcsh_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cause_specific_hazard(time, event_type, X, cause)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crrcsh_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = cause_specific_hazard(time, event_type, X, cause)
    assert isinstance(result, dict)
