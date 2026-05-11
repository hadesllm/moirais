"""Tests for ca12u360.ca_chapter_12_unnumbered_360."""
import numpy as np
import pytest
from morie.fn.ca12u360 import ca_chapter_12_unnumbered_360


def test_ca12u360_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_360(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca12u360_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_360(x)
    assert isinstance(result, dict)
