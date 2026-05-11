"""Tests for wilcox13u1262.wilcox_chapter_13_unnumbered_1262."""
import numpy as np
import pytest
from morie.fn.wilcox13u1262 import wilcox_chapter_13_unnumbered_1262


def test_wilcox13u1262_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1262(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1262_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1262(x)
    assert isinstance(result, dict)
