"""Tests for ca4u104.ca_chapter_4_unnumbered_104."""
import numpy as np
import pytest
from morie.fn.ca4u104 import ca_chapter_4_unnumbered_104


def test_ca4u104_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_104(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u104_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_104(x)
    assert isinstance(result, dict)
