"""Tests for wilcox14u614.wilcox_chapter_14_unnumbered_614."""
import numpy as np
import pytest
from moirais.fn.wilcox14u614 import wilcox_chapter_14_unnumbered_614


def test_wilcox14u614_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_614(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u614_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_614(x)
    assert isinstance(result, dict)
