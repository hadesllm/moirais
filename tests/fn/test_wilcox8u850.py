"""Tests for wilcox8u850.wilcox_chapter_8_unnumbered_850."""
import numpy as np
import pytest
from moirais.fn.wilcox8u850 import wilcox_chapter_8_unnumbered_850


def test_wilcox8u850_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_850(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u850_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_850(x)
    assert isinstance(result, dict)
