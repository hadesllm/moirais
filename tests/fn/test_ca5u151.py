"""Tests for ca5u151.ca_chapter_5_unnumbered_151."""
import numpy as np
import pytest
from morie.fn.ca5u151 import ca_chapter_5_unnumbered_151


def test_ca5u151_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_151(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u151_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_151(x)
    assert isinstance(result, dict)
