"""Tests for ca12u364.ca_chapter_12_unnumbered_364."""
import numpy as np
import pytest
from moirais.fn.ca12u364 import ca_chapter_12_unnumbered_364


def test_ca12u364_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_364(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca12u364_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_364(x)
    assert isinstance(result, dict)
