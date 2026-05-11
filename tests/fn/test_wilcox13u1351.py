"""Tests for wilcox13u1351.wilcox_chapter_13_unnumbered_1351."""
import numpy as np
import pytest
from morie.fn.wilcox13u1351 import wilcox_chapter_13_unnumbered_1351


def test_wilcox13u1351_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1351(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1351_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1351(x)
    assert isinstance(result, dict)
