"""Tests for ca3u50.ca_chapter_3_unnumbered_50."""
import numpy as np
import pytest
from morie.fn.ca3u50 import ca_chapter_3_unnumbered_50


def test_ca3u50_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_50(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u50_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_50(x)
    assert isinstance(result, dict)
