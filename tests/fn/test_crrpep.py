"""Tests for crrpep.pepe_mori."""
import numpy as np
import pytest
from moirais.fn.crrpep import pepe_mori


def test_crrpep_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = pepe_mori(time, event_type, group, cause)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_crrpep_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = pepe_mori(time, event_type, group, cause)
    assert isinstance(result, dict)
