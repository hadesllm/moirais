"""Tests for hetlc.heterozygosity_locus."""
import numpy as np
import pytest
from moirais.fn.hetlc import heterozygosity_locus


def test_hetlc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = heterozygosity_locus(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_hetlc_edge():
    """Test edge cases."""
    result = heterozygosity_locus(np.array([42.0]))
    assert result['n'] == 1
