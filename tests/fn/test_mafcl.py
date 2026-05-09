"""Tests for mafcl.maf_calculation."""
import numpy as np
import pytest
from moirais.fn.mafcl import maf_calculation


def test_mafcl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = maf_calculation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mafcl_edge():
    """Test edge cases."""
    result = maf_calculation(np.array([42.0]))
    assert result['n'] == 1
