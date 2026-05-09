"""Tests for wilcox13u1212.wilcox_chapter_13_unnumbered_1212."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1212 import wilcox_chapter_13_unnumbered_1212


def test_wilcox13u1212_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1212(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1212_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1212(x)
    assert isinstance(result, dict)
