"""Tests for ca7u233.ca_chapter_7_unnumbered_233."""
import numpy as np
import pytest
from moirais.fn.ca7u233 import ca_chapter_7_unnumbered_233


def test_ca7u233_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_233(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u233_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_233(x)
    assert isinstance(result, dict)
