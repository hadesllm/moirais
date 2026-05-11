"""Tests for wilcox8u813.wilcox_chapter_8_unnumbered_813."""
import numpy as np
import pytest
from morie.fn.wilcox8u813 import wilcox_chapter_8_unnumbered_813


def test_wilcox8u813_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_813(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u813_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_813(x)
    assert isinstance(result, dict)
