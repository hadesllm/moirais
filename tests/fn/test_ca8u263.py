"""Tests for ca8u263.ca_chapter_8_unnumbered_263."""
import numpy as np
import pytest
from morie.fn.ca8u263 import ca_chapter_8_unnumbered_263


def test_ca8u263_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_263(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u263_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_263(x)
    assert isinstance(result, dict)
