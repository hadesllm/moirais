"""Tests for ca6u210.ca_chapter_6_unnumbered_210."""
import numpy as np
import pytest
from morie.fn.ca6u210 import ca_chapter_6_unnumbered_210


def test_ca6u210_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_210(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u210_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_210(x)
    assert isinstance(result, dict)
