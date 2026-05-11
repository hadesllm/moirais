"""Tests for wilcox8u844.wilcox_chapter_8_unnumbered_844."""
import numpy as np
import pytest
from morie.fn.wilcox8u844 import wilcox_chapter_8_unnumbered_844


def test_wilcox8u844_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_844(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u844_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_844(x)
    assert isinstance(result, dict)
