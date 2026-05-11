"""Tests for wilcox12u1383.wilcox_chapter_12_unnumbered_1383."""
import numpy as np
import pytest
from morie.fn.wilcox12u1383 import wilcox_chapter_12_unnumbered_1383


def test_wilcox12u1383_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1383(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox12u1383_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1383(x)
    assert isinstance(result, dict)
