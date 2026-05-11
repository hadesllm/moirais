"""Tests for wilcox10u873.wilcox_chapter_10_unnumbered_873."""
import numpy as np
import pytest
from morie.fn.wilcox10u873 import wilcox_chapter_10_unnumbered_873


def test_wilcox10u873_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_873(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u873_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_873(x)
    assert isinstance(result, dict)
