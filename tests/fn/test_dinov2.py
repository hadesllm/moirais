"""Tests for dinov2.dino_v2_repr."""
import numpy as np
import pytest
from moirais.fn.dinov2 import dino_v2_repr


def test_dinov2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = dino_v2_repr(x, student, teacher, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dinov2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = dino_v2_repr(x, student, teacher, tau)
    assert isinstance(result, dict)
