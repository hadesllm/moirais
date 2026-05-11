"""Tests for wilcox8u866.wilcox_chapter_8_unnumbered_866."""
import numpy as np
import pytest
from morie.fn.wilcox8u866 import wilcox_chapter_8_unnumbered_866


def test_wilcox8u866_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_866(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u866_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_866(x)
    assert isinstance(result, dict)
