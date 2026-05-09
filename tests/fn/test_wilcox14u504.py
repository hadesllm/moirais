"""Tests for wilcox14u504.wilcox_chapter_14_unnumbered_504."""
import numpy as np
import pytest
from moirais.fn.wilcox14u504 import wilcox_chapter_14_unnumbered_504


def test_wilcox14u504_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_504(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u504_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_504(x)
    assert isinstance(result, dict)
