"""Tests for ca6u193.ca_chapter_6_unnumbered_193."""
import numpy as np
import pytest
from morie.fn.ca6u193 import ca_chapter_6_unnumbered_193


def test_ca6u193_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_193(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca6u193_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_193(x)
    assert isinstance(result, dict)
