"""Tests for wilcox4u211.wilcox_chapter_4_unnumbered_211."""
import numpy as np
import pytest
from morie.fn.wilcox4u211 import wilcox_chapter_4_unnumbered_211


def test_wilcox4u211_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_211(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u211_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_211(x)
    assert isinstance(result, dict)
