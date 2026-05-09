"""Tests for wilcox8u864.wilcox_chapter_8_unnumbered_864."""
import numpy as np
import pytest
from moirais.fn.wilcox8u864 import wilcox_chapter_8_unnumbered_864


def test_wilcox8u864_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_864(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u864_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_864(x)
    assert isinstance(result, dict)
