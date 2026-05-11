"""Tests for agrsc.agreement_score_matrix."""
import numpy as np
import pytest
from morie.fn.agrsc import agreement_score_matrix


def test_agrsc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = agreement_score_matrix(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_agrsc_edge():
    """Test edge cases."""
    result = agreement_score_matrix(np.array([42.0]))
    assert result['n'] == 1
