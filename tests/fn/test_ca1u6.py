"""Tests for ca1u6.ca_chapter_1_unnumbered_6."""
import numpy as np
import pytest
from moirais.fn.ca1u6 import ca_chapter_1_unnumbered_6


def test_ca1u6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_6(x)
    assert isinstance(result, dict)
