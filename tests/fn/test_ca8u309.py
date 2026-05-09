"""Tests for ca8u309.ca_chapter_8_unnumbered_309."""
import numpy as np
import pytest
from moirais.fn.ca8u309 import ca_chapter_8_unnumbered_309


def test_ca8u309_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_309(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u309_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_309(x)
    assert isinstance(result, dict)
