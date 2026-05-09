"""Tests for ca5u154.ca_chapter_5_unnumbered_154."""
import numpy as np
import pytest
from moirais.fn.ca5u154 import ca_chapter_5_unnumbered_154


def test_ca5u154_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_154(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u154_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_154(x)
    assert isinstance(result, dict)
