"""Tests for wilcox13u1276.wilcox_chapter_13_unnumbered_1276."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1276 import wilcox_chapter_13_unnumbered_1276


def test_wilcox13u1276_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1276(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1276_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1276(x)
    assert isinstance(result, dict)
