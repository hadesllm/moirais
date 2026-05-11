"""Tests for dinmlt.dino_multicrop."""
import numpy as np
import pytest
from morie.fn.dinmlt import dino_multicrop


def test_dinmlt_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    global_size = 100
    local_size = 100
    result = dino_multicrop(image, global_size, local_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dinmlt_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    global_size = 100
    local_size = 100
    result = dino_multicrop(image, global_size, local_size)
    assert isinstance(result, dict)
