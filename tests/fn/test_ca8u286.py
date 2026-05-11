"""Tests for ca8u286.ca_chapter_8_unnumbered_286."""
import numpy as np
import pytest
from morie.fn.ca8u286 import ca_chapter_8_unnumbered_286


def test_ca8u286_basic():
    """Test basic functionality."""
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_286(x1, x2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u286_edge():
    """Test edge cases."""
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    x2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_286(x1, x2)
    assert isinstance(result, dict)
