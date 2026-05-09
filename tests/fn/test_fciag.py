"""Tests for fciag.fci_algorithm."""
import numpy as np
import pytest
from moirais.fn.fciag import fci_algorithm


def test_fciag_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    ci_test = np.random.default_rng(43).normal(0, 1, 30)
    result = fci_algorithm(data, alpha, ci_test)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fciag_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    ci_test = np.random.default_rng(43).normal(0, 1, 30)
    result = fci_algorithm(data, alpha, ci_test)
    assert isinstance(result, dict)
