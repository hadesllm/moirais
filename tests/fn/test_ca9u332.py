"""Tests for ca9u332.ca_chapter_9_unnumbered_332."""
import numpy as np
import pytest
from morie.fn.ca9u332 import ca_chapter_9_unnumbered_332


def test_ca9u332_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_332(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca9u332_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_332(x)
    assert isinstance(result, dict)
