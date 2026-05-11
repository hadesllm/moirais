"""Tests for ca9u328.ca_chapter_9_unnumbered_328."""
import numpy as np
import pytest
from morie.fn.ca9u328 import ca_chapter_9_unnumbered_328


def test_ca9u328_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_328(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca9u328_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_328(x)
    assert isinstance(result, dict)
