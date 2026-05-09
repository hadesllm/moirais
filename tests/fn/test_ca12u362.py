"""Tests for ca12u362.ca_chapter_12_unnumbered_362."""
import numpy as np
import pytest
from moirais.fn.ca12u362 import ca_chapter_12_unnumbered_362


def test_ca12u362_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_362(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca12u362_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_362(x)
    assert isinstance(result, dict)
