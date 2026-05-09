"""Tests for fzqnt.fauzi_kernel_quantile_asymptotic."""
import numpy as np
import pytest
from moirais.fn.fzqnt import fauzi_kernel_quantile_asymptotic


def test_fzqnt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_kernel_quantile_asymptotic(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzqnt_edge():
    """Test edge cases."""
    result = fauzi_kernel_quantile_asymptotic(np.array([42.0]))
    assert result['n'] == 1
