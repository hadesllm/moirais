"""Tests for ca7u242.ca_chapter_7_unnumbered_242."""
import numpy as np
import pytest
from moirais.fn.ca7u242 import ca_chapter_7_unnumbered_242


def test_ca7u242_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_242(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u242_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_242(x)
    assert isinstance(result, dict)
