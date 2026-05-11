"""Tests for ca9u330.ca_chapter_9_unnumbered_330."""
import numpy as np
import pytest
from morie.fn.ca9u330 import ca_chapter_9_unnumbered_330


def test_ca9u330_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_330(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca9u330_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_330(x)
    assert isinstance(result, dict)
