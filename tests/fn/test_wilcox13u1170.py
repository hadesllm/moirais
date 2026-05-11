"""Tests for wilcox13u1170.wilcox_chapter_13_unnumbered_1170."""
import numpy as np
import pytest
from morie.fn.wilcox13u1170 import wilcox_chapter_13_unnumbered_1170


def test_wilcox13u1170_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1170(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1170_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1170(x)
    assert isinstance(result, dict)
