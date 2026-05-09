"""Tests for ca7u236.ca_chapter_7_unnumbered_236."""
import numpy as np
import pytest
from moirais.fn.ca7u236 import ca_chapter_7_unnumbered_236


def test_ca7u236_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_236(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u236_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_236(x)
    assert isinstance(result, dict)
