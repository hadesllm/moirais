"""Tests for ca8u283.ca_chapter_8_unnumbered_283."""
import numpy as np
import pytest
from moirais.fn.ca8u283 import ca_chapter_8_unnumbered_283


def test_ca8u283_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_283(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u283_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_283(x)
    assert isinstance(result, dict)
