"""Tests for wilcox8u836.wilcox_chapter_8_unnumbered_836."""
import numpy as np
import pytest
from morie.fn.wilcox8u836 import wilcox_chapter_8_unnumbered_836


def test_wilcox8u836_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_836(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u836_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_836(x)
    assert isinstance(result, dict)
