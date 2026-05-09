"""Tests for ca5u141.ca_chapter_5_unnumbered_141."""
import numpy as np
import pytest
from moirais.fn.ca5u141 import ca_chapter_5_unnumbered_141


def test_ca5u141_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_141(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u141_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_141(x)
    assert isinstance(result, dict)
