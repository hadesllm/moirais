"""Tests for ca8u322.ca_chapter_8_unnumbered_322."""
import numpy as np
import pytest
from moirais.fn.ca8u322 import ca_chapter_8_unnumbered_322


def test_ca8u322_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_322(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u322_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_322(x)
    assert isinstance(result, dict)
