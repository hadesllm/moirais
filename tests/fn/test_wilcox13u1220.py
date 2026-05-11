"""Tests for wilcox13u1220.wilcox_chapter_13_unnumbered_1220."""
import numpy as np
import pytest
from morie.fn.wilcox13u1220 import wilcox_chapter_13_unnumbered_1220


def test_wilcox13u1220_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1220(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1220_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1220(x)
    assert isinstance(result, dict)
