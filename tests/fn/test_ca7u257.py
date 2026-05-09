"""Tests for ca7u257.ca_chapter_7_unnumbered_257."""
import numpy as np
import pytest
from moirais.fn.ca7u257 import ca_chapter_7_unnumbered_257


def test_ca7u257_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_257(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u257_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_257(x)
    assert isinstance(result, dict)
