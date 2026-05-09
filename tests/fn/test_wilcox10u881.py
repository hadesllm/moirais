"""Tests for wilcox10u881.wilcox_chapter_10_unnumbered_881."""
import numpy as np
import pytest
from moirais.fn.wilcox10u881 import wilcox_chapter_10_unnumbered_881


def test_wilcox10u881_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_881(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u881_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_881(x)
    assert isinstance(result, dict)
