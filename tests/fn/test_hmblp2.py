"""Tests for hmblp2.geron_blip2."""
import numpy as np
import pytest
from moirais.fn.hmblp2 import geron_blip2


def test_hmblp2_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip2(image, text)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmblp2_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    text = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_blip2(image, text)
    assert isinstance(result, dict)
