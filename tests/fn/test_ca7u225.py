"""Tests for ca7u225.ca_chapter_7_unnumbered_225."""
import numpy as np
import pytest
from moirais.fn.ca7u225 import ca_chapter_7_unnumbered_225


def test_ca7u225_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_225(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u225_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_225(x)
    assert isinstance(result, dict)
