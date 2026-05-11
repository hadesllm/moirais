"""Tests for fzwlc.fauzi_smoothed_wilcoxon."""
import numpy as np
import pytest
from morie.fn.fzwlc import fauzi_smoothed_wilcoxon


def test_fzwlc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_smoothed_wilcoxon(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzwlc_edge():
    """Test edge cases."""
    result = fauzi_smoothed_wilcoxon(np.array([42.0]))
    assert result['n'] == 1
