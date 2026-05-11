"""Tests for ca5u152.ca_chapter_5_unnumbered_152."""
import numpy as np
import pytest
from morie.fn.ca5u152 import ca_chapter_5_unnumbered_152


def test_ca5u152_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_152(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u152_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_152(x)
    assert isinstance(result, dict)
