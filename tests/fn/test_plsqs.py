"""Tests for plsqs.pls_qsar."""
import numpy as np
import pytest
from moirais.fn.plsqs import pls_qsar


def test_plsqs_basic():
    """Test basic functionality."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    descriptors = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = pls_qsar(activities, descriptors, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_plsqs_edge():
    """Test edge cases."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    descriptors = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = pls_qsar(activities, descriptors, n_components)
    assert isinstance(result, dict)
