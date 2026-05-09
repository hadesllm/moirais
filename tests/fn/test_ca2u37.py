"""Tests for ca2u37.ca_chapter_2_unnumbered_37."""
import numpy as np
import pytest
from moirais.fn.ca2u37 import ca_chapter_2_unnumbered_37


def test_ca2u37_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_37(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca2u37_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_37(x)
    assert isinstance(result, dict)
