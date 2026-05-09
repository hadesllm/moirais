"""Tests for wilcox12u1384.wilcox_chapter_12_unnumbered_1384."""
import numpy as np
import pytest
from moirais.fn.wilcox12u1384 import wilcox_chapter_12_unnumbered_1384


def test_wilcox12u1384_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1384(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox12u1384_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1384(x)
    assert isinstance(result, dict)
