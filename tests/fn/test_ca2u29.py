"""Tests for ca2u29.ca_chapter_2_unnumbered_29."""
import numpy as np
import pytest
from morie.fn.ca2u29 import ca_chapter_2_unnumbered_29


def test_ca2u29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_29(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2u29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_29(x)
    assert isinstance(result, dict)
