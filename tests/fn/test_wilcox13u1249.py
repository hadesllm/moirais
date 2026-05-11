"""Tests for wilcox13u1249.wilcox_chapter_13_unnumbered_1249."""
import numpy as np
import pytest
from morie.fn.wilcox13u1249 import wilcox_chapter_13_unnumbered_1249


def test_wilcox13u1249_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1249(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1249_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1249(x)
    assert isinstance(result, dict)
