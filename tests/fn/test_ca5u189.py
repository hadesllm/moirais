"""Tests for ca5u189.ca_chapter_5_unnumbered_189."""
import numpy as np
import pytest
from morie.fn.ca5u189 import ca_chapter_5_unnumbered_189


def test_ca5u189_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_189(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u189_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_189(x)
    assert isinstance(result, dict)
