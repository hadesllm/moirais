"""Tests for agdsts.alphazero_distill_student."""
import numpy as np
import pytest
from morie.fn.agdsts import alphazero_distill_student


def test_agdsts_basic():
    """Test basic functionality."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_distill_student(teacher, student, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agdsts_edge():
    """Test edge cases."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_distill_student(teacher, student, data)
    assert isinstance(result, dict)
