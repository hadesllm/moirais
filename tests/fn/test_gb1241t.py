"""Tests for gb1241t.gibbons_concordance_w_ties."""
import numpy as np
import pytest
from moirais.fn.gb1241t import gibbons_concordance_w_ties


def test_gb1241t_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_concordance_w_ties(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb1241t_edge():
    """Test edge cases."""
    result = gibbons_concordance_w_ties(np.array([42.0]))
    assert result['n'] == 1
