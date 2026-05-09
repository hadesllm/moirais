"""Tests for gb_cq.gibbons_cramers_contingency."""
import numpy as np
import pytest
from moirais.fn.gb_cq import gibbons_cramers_contingency


def test_gb_cq_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_cramers_contingency(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_cq_edge():
    """Test edge cases."""
    result = gibbons_cramers_contingency(np.array([42.0]))
    assert result['n'] == 1
