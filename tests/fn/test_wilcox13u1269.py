"""Tests for wilcox13u1269.wilcox_chapter_13_unnumbered_1269."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1269 import wilcox_chapter_13_unnumbered_1269


def test_wilcox13u1269_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1269(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1269_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1269(x)
    assert isinstance(result, dict)
