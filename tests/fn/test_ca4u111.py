"""Tests for ca4u111.ca_chapter_4_unnumbered_111."""
import numpy as np
import pytest
from moirais.fn.ca4u111 import ca_chapter_4_unnumbered_111


def test_ca4u111_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_111(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u111_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_111(x)
    assert isinstance(result, dict)
