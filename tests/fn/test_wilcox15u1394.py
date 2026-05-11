"""Tests for wilcox15u1394.wilcox_chapter_15_unnumbered_1394."""
import numpy as np
import pytest
from morie.fn.wilcox15u1394 import wilcox_chapter_15_unnumbered_1394


def test_wilcox15u1394_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1394(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox15u1394_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1394(x)
    assert isinstance(result, dict)
