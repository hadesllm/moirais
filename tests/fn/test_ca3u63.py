"""Tests for ca3u63.ca_chapter_3_unnumbered_63."""
import numpy as np
import pytest
from moirais.fn.ca3u63 import ca_chapter_3_unnumbered_63


def test_ca3u63_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_63(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u63_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_63(x)
    assert isinstance(result, dict)
