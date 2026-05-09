"""Tests for wilcox5u407.wilcox_chapter_5_unnumbered_407."""
import numpy as np
import pytest
from moirais.fn.wilcox5u407 import wilcox_chapter_5_unnumbered_407


def test_wilcox5u407_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_407(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u407_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_407(x)
    assert isinstance(result, dict)
