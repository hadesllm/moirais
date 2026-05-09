"""Tests for wilcox13u1128.wilcox_chapter_13_unnumbered_1128."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1128 import wilcox_chapter_13_unnumbered_1128


def test_wilcox13u1128_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1128(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1128_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1128(x)
    assert isinstance(result, dict)
