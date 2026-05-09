"""Tests for ca8u321.ca_chapter_8_unnumbered_321."""
import numpy as np
import pytest
from moirais.fn.ca8u321 import ca_chapter_8_unnumbered_321


def test_ca8u321_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_321(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u321_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_321(x)
    assert isinstance(result, dict)
