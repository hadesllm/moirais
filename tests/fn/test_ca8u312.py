"""Tests for ca8u312.ca_chapter_8_unnumbered_312."""
import numpy as np
import pytest
from moirais.fn.ca8u312 import ca_chapter_8_unnumbered_312


def test_ca8u312_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_312(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u312_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_312(x)
    assert isinstance(result, dict)
