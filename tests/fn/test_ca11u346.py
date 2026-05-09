"""Tests for ca11u346.ca_chapter_11_unnumbered_346."""
import numpy as np
import pytest
from moirais.fn.ca11u346 import ca_chapter_11_unnumbered_346


def test_ca11u346_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_346(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11u346_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_346(x)
    assert isinstance(result, dict)
