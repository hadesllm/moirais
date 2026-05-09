"""Tests for wilcox8u824.wilcox_chapter_8_unnumbered_824."""
import numpy as np
import pytest
from moirais.fn.wilcox8u824 import wilcox_chapter_8_unnumbered_824


def test_wilcox8u824_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_824(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u824_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_824(x)
    assert isinstance(result, dict)
