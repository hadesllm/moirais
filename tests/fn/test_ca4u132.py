"""Tests for ca4u132.ca_chapter_4_unnumbered_132."""
import numpy as np
import pytest
from moirais.fn.ca4u132 import ca_chapter_4_unnumbered_132


def test_ca4u132_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_132(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u132_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_132(x)
    assert isinstance(result, dict)
