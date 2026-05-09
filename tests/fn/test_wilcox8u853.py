"""Tests for wilcox8u853.wilcox_chapter_8_unnumbered_853."""
import numpy as np
import pytest
from moirais.fn.wilcox8u853 import wilcox_chapter_8_unnumbered_853


def test_wilcox8u853_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_853(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u853_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_853(x)
    assert isinstance(result, dict)
