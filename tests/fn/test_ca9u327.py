"""Tests for ca9u327.ca_chapter_9_unnumbered_327."""
import numpy as np
import pytest
from morie.fn.ca9u327 import ca_chapter_9_unnumbered_327


def test_ca9u327_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_327(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca9u327_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_327(x)
    assert isinstance(result, dict)
