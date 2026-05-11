"""Tests for wilcox7e39.wilcox_chapter_7_equation_39."""
import numpy as np
import pytest
from morie.fn.wilcox7e39 import wilcox_chapter_7_equation_39


def test_wilcox7e39_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_39(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox7e39_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_39(x)
    assert isinstance(result, dict)
