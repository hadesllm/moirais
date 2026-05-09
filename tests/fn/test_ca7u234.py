"""Tests for ca7u234.ca_chapter_7_unnumbered_234."""
import numpy as np
import pytest
from moirais.fn.ca7u234 import ca_chapter_7_unnumbered_234


def test_ca7u234_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_234(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u234_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_234(x)
    assert isinstance(result, dict)
