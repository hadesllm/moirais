"""Tests for ca10u335.ca_chapter_10_unnumbered_335."""
import numpy as np
import pytest
from moirais.fn.ca10u335 import ca_chapter_10_unnumbered_335


def test_ca10u335_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_335(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca10u335_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_335(x)
    assert isinstance(result, dict)
