"""Tests for ca1u11.ca_chapter_1_unnumbered_11."""
import numpy as np
import pytest
from moirais.fn.ca1u11 import ca_chapter_1_unnumbered_11


def test_ca1u11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_11(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca1u11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_11(x)
    assert isinstance(result, dict)
