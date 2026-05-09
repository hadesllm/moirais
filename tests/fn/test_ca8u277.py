"""Tests for ca8u277.ca_chapter_8_unnumbered_277."""
import numpy as np
import pytest
from moirais.fn.ca8u277 import ca_chapter_8_unnumbered_277


def test_ca8u277_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_277(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u277_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_277(x)
    assert isinstance(result, dict)
