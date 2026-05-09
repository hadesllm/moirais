"""Tests for opnclp.open_clip."""
import numpy as np
import pytest
from moirais.fn.opnclp import open_clip


def test_opnclp_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    batch = np.random.default_rng(42).normal(0, 1, 100)
    result = open_clip(images, texts, batch)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_opnclp_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    texts = np.random.default_rng(42).normal(0, 1, 100)
    batch = np.random.default_rng(42).normal(0, 1, 100)
    result = open_clip(images, texts, batch)
    assert isinstance(result, dict)
