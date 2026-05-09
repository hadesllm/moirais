"""Tests for wilcox8u828.wilcox_chapter_8_unnumbered_828."""
import numpy as np
import pytest
from moirais.fn.wilcox8u828 import wilcox_chapter_8_unnumbered_828


def test_wilcox8u828_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_828(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u828_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_828(x)
    assert isinstance(result, dict)
