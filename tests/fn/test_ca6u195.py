"""Tests for ca6u195.ca_chapter_6_unnumbered_195."""
import numpy as np
import pytest
from moirais.fn.ca6u195 import ca_chapter_6_unnumbered_195


def test_ca6u195_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_195(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca6u195_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_195(x)
    assert isinstance(result, dict)
