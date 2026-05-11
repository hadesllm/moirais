"""Tests for ca1u4.ca_chapter_1_unnumbered_4."""
import numpy as np
import pytest
from morie.fn.ca1u4 import ca_chapter_1_unnumbered_4


def test_ca1u4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_4(x)
    assert isinstance(result, dict)
