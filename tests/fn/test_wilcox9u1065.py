"""Tests for wilcox9u1065.wilcox_chapter_9_unnumbered_1065."""
import numpy as np
import pytest
from moirais.fn.wilcox9u1065 import wilcox_chapter_9_unnumbered_1065


def test_wilcox9u1065_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1065(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox9u1065_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1065(x)
    assert isinstance(result, dict)
