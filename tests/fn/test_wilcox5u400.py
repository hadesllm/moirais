"""Tests for wilcox5u400.wilcox_chapter_5_unnumbered_400."""
import numpy as np
import pytest
from moirais.fn.wilcox5u400 import wilcox_chapter_5_unnumbered_400


def test_wilcox5u400_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_400(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u400_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_400(x)
    assert isinstance(result, dict)
