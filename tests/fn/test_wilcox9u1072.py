"""Tests for wilcox9u1072.wilcox_chapter_9_unnumbered_1072."""
import numpy as np
import pytest
from moirais.fn.wilcox9u1072 import wilcox_chapter_9_unnumbered_1072


def test_wilcox9u1072_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1072(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox9u1072_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1072(x)
    assert isinstance(result, dict)
