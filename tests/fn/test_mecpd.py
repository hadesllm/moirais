"""Tests for mecpd.markov_equivalence_class."""
import numpy as np
import pytest
from morie.fn.mecpd import markov_equivalence_class


def test_mecpd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = markov_equivalence_class(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mecpd_edge():
    """Test edge cases."""
    result = markov_equivalence_class(np.array([42.0]))
    assert result['n'] == 1
