"""Tests for gh_py_univ_seq.ghosal_py_universal_sequence."""
import numpy as np
import pytest
from moirais.fn.gh_py_univ_seq import ghosal_py_universal_sequence


def test_gh_py_univ_seq_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_py_universal_sequence(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_py_univ_seq_edge():
    """Test edge cases."""
    result = ghosal_py_universal_sequence(np.array([42.0]))
    assert result['n'] == 1
