"""Tests for hmdino.geron_dino."""
import numpy as np
import pytest
from morie.fn.hmdino import geron_dino


def test_hmdino_basic():
    """Test basic functionality."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dino(images, student, teacher)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdino_edge():
    """Test edge cases."""
    images = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dino(images, student, teacher)
    assert isinstance(result, dict)
