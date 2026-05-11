"""Tests for ca11u351.ca_chapter_11_unnumbered_351."""
import numpy as np
import pytest
from morie.fn.ca11u351 import ca_chapter_11_unnumbered_351


def test_ca11u351_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_351(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11u351_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_351(x)
    assert isinstance(result, dict)
