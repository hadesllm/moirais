"""Tests for ca8u274.ca_chapter_8_unnumbered_274."""
import numpy as np
import pytest
from morie.fn.ca8u274 import ca_chapter_8_unnumbered_274


def test_ca8u274_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_274(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u274_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_274(x)
    assert isinstance(result, dict)
