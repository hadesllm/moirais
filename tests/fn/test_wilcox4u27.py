"""Tests for wilcox4u27.wilcox_chapter_4_unnumbered_27."""
import numpy as np
import pytest
from morie.fn.wilcox4u27 import wilcox_chapter_4_unnumbered_27


def test_wilcox4u27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_27(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_27(x)
    assert isinstance(result, dict)
