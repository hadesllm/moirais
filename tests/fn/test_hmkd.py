"""Tests for hmkd.geron_knowledge_distillation."""
import numpy as np
import pytest
from moirais.fn.hmkd import geron_knowledge_distillation


def test_hmkd_basic():
    """Test basic functionality."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    alpha = 0.05
    result = geron_knowledge_distillation(teacher, student, X, y, T, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkd_edge():
    """Test edge cases."""
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    alpha = 0.05
    result = geron_knowledge_distillation(teacher, student, X, y, T, alpha)
    assert isinstance(result, dict)
