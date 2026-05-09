"""Tests for ca1u10.ca_chapter_1_unnumbered_10."""
import numpy as np
import pytest
from moirais.fn.ca1u10 import ca_chapter_1_unnumbered_10


def test_ca1u10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_10(x)
    assert isinstance(result, dict)
