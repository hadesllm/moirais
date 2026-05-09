"""Tests for gh_c8_9.ghosal_markov_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c8_9 import ghosal_markov_crt


def test_gh_c8_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_markov_crt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_9_edge():
    """Test edge cases."""
    result = ghosal_markov_crt(np.array([42.0]))
    assert result['n'] == 1
