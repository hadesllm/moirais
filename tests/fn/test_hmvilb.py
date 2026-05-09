"""Tests for hmvilb.geron_vilbert."""
import numpy as np
import pytest
from moirais.fn.hmvilb import geron_vilbert


def test_hmvilb_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vilbert(image, text)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvilb_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vilbert(image, text)
    assert isinstance(result, dict)
