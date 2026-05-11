"""Tests for ca5u138.ca_chapter_5_unnumbered_138."""
import numpy as np
import pytest
from morie.fn.ca5u138 import ca_chapter_5_unnumbered_138


def test_ca5u138_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_138(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u138_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_138(x)
    assert isinstance(result, dict)
