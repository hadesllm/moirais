"""Tests for ca7u235.ca_chapter_7_unnumbered_235."""
import numpy as np
import pytest
from moirais.fn.ca7u235 import ca_chapter_7_unnumbered_235


def test_ca7u235_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_235(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u235_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_235(x)
    assert isinstance(result, dict)
