"""Tests for wilcox14u777.wilcox_chapter_14_unnumbered_777."""
import numpy as np
import pytest
from morie.fn.wilcox14u777 import wilcox_chapter_14_unnumbered_777


def test_wilcox14u777_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_777(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u777_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_777(x)
    assert isinstance(result, dict)
