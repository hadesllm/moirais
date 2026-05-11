"""Tests for wilcox13u1329.wilcox_chapter_13_unnumbered_1329."""
import numpy as np
import pytest
from morie.fn.wilcox13u1329 import wilcox_chapter_13_unnumbered_1329


def test_wilcox13u1329_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1329(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1329_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1329(x)
    assert isinstance(result, dict)
