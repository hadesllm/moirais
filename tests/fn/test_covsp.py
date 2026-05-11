"""Tests for covsp.one_sample_coverage."""
import numpy as np
import pytest
from morie.fn.covsp import one_sample_coverage


def test_covsp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = one_sample_coverage(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_covsp_edge():
    """Test edge cases."""
    result = one_sample_coverage(np.array([42.0]))
    assert result['n'] == 1
