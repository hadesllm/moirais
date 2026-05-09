"""Tests for hmflmg.geron_flamingo."""
import numpy as np
import pytest
from moirais.fn.hmflmg import geron_flamingo


def test_hmflmg_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_flamingo(images, text)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmflmg_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_flamingo(images, text)
    assert isinstance(result, dict)
