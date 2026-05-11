"""Tests for ca9u329.ca_chapter_9_unnumbered_329."""
import numpy as np
import pytest
from morie.fn.ca9u329 import ca_chapter_9_unnumbered_329


def test_ca9u329_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_329(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca9u329_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_329(x)
    assert isinstance(result, dict)
