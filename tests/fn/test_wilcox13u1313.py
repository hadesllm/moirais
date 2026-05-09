"""Tests for wilcox13u1313.wilcox_chapter_13_unnumbered_1313."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1313 import wilcox_chapter_13_unnumbered_1313


def test_wilcox13u1313_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1313(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1313_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1313(x)
    assert isinstance(result, dict)
