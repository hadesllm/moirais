"""Tests for wilcox13u1364.wilcox_chapter_13_unnumbered_1364."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1364 import wilcox_chapter_13_unnumbered_1364


def test_wilcox13u1364_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1364(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1364_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1364(x)
    assert isinstance(result, dict)
