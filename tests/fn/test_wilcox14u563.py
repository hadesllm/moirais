"""Tests for wilcox14u563.wilcox_chapter_14_unnumbered_563."""
import numpy as np
import pytest
from morie.fn.wilcox14u563 import wilcox_chapter_14_unnumbered_563


def test_wilcox14u563_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_563(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u563_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_563(x)
    assert isinstance(result, dict)
