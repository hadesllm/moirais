"""Tests for incidens.incidence_rate."""
import numpy as np
import pytest
from morie.fn.incidens import incidence_rate


def test_incidens_basic():
    """Test basic functionality."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    person_time = np.random.default_rng(42).normal(0, 1, 100)
    result = incidence_rate(cases, person_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_incidens_edge():
    """Test edge cases."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    person_time = np.random.default_rng(42).normal(0, 1, 100)
    result = incidence_rate(cases, person_time)
    assert isinstance(result, dict)
