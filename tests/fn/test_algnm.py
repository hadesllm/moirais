"""Tests for algnm.party_alignment."""
import numpy as np
import pytest
from moirais.fn.algnm import party_alignment


def test_algnm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = party_alignment(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_algnm_edge():
    """Test edge cases."""
    result = party_alignment(np.array([42.0]))
    assert result['n'] == 1
