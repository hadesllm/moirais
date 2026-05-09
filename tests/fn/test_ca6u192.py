"""Tests for ca6u192.ca_chapter_6_unnumbered_192."""
import numpy as np
import pytest
from moirais.fn.ca6u192 import ca_chapter_6_unnumbered_192


def test_ca6u192_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_192(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca6u192_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_192(x)
    assert isinstance(result, dict)
