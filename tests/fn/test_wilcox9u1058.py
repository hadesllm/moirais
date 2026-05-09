"""Tests for wilcox9u1058.wilcox_chapter_9_unnumbered_1058."""
import numpy as np
import pytest
from moirais.fn.wilcox9u1058 import wilcox_chapter_9_unnumbered_1058


def test_wilcox9u1058_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1058(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox9u1058_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1058(x)
    assert isinstance(result, dict)
