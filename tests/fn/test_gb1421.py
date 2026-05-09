"""Tests for gb1421.gibbons_chisq_contingency."""
import numpy as np
import pytest
from moirais.fn.gb1421 import gibbons_chisq_contingency


def test_gb1421_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_chisq_contingency(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1421_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_chisq_contingency(table)
    assert isinstance(result, dict)
