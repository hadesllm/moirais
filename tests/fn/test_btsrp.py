"""Tests for btsrp.bootstrap_ci."""
import numpy as np
import pytest
from moirais.fn.btsrp import bootstrap_ci


def test_btsrp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bootstrap_ci(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_btsrp_edge():
    """Test edge cases."""
    result = bootstrap_ci(np.array([42.0]))
    assert result['n'] == 1
