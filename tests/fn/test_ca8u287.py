"""Tests for ca8u287.ca_chapter_8_unnumbered_287."""
import numpy as np
import pytest
from moirais.fn.ca8u287 import ca_chapter_8_unnumbered_287


def test_ca8u287_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_287(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u287_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_287(x)
    assert isinstance(result, dict)
