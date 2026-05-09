"""Tests for hmdbrt.geron_distilbert."""
import numpy as np
import pytest
from moirais.fn.hmdbrt import geron_distilbert


def test_hmdbrt_basic():
    """Test basic functionality."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_distilbert(teacher, student, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdbrt_edge():
    """Test edge cases."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_distilbert(teacher, student, X)
    assert isinstance(result, dict)
