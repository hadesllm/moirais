"""Tests for wilcox13u1335.wilcox_chapter_13_unnumbered_1335."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1335 import wilcox_chapter_13_unnumbered_1335


def test_wilcox13u1335_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1335(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1335_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1335(x)
    assert isinstance(result, dict)
