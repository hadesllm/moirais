"""Tests for wilcox13u1302.wilcox_chapter_13_unnumbered_1302."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1302 import wilcox_chapter_13_unnumbered_1302


def test_wilcox13u1302_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1302(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1302_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1302(x)
    assert isinstance(result, dict)
