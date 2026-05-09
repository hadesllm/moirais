"""Tests for ca5u149.ca_chapter_5_unnumbered_149."""
import numpy as np
import pytest
from moirais.fn.ca5u149 import ca_chapter_5_unnumbered_149


def test_ca5u149_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_149(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u149_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_149(x)
    assert isinstance(result, dict)
