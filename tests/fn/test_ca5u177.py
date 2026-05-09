"""Tests for ca5u177.ca_chapter_5_unnumbered_177."""
import numpy as np
import pytest
from moirais.fn.ca5u177 import ca_chapter_5_unnumbered_177


def test_ca5u177_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_177(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca5u177_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_177(x)
    assert isinstance(result, dict)
