"""Tests for ghhbp.ghosal_hierarchical_bayes."""
import numpy as np
import pytest
from morie.fn.ghhbp import ghosal_hierarchical_bayes


def test_ghhbp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_hierarchical_bayes(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghhbp_edge():
    """Test edge cases."""
    result = ghosal_hierarchical_bayes(np.array([42.0]))
    assert result['n'] == 1
