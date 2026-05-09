"""Tests for ca5u180.ca_chapter_5_unnumbered_180."""
import numpy as np
import pytest
from moirais.fn.ca5u180 import ca_chapter_5_unnumbered_180


def test_ca5u180_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_180(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u180_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_180(x)
    assert isinstance(result, dict)
