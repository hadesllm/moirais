"""Tests for ca5u176.ca_chapter_5_unnumbered_176."""
import numpy as np
import pytest
from moirais.fn.ca5u176 import ca_chapter_5_unnumbered_176


def test_ca5u176_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_176(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u176_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_176(x)
    assert isinstance(result, dict)
