"""Tests for gh_emp_bayes.ghosal_empirical_bayes_np."""
import numpy as np
import pytest
from moirais.fn.gh_emp_bayes import ghosal_empirical_bayes_np


def test_gh_emp_bayes_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_empirical_bayes_np(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_emp_bayes_edge():
    """Test edge cases."""
    result = ghosal_empirical_bayes_np(np.array([42.0]))
    assert result['n'] == 1
