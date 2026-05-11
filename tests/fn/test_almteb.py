"""Tests for almteb.alammar_mteb_benchmark_score."""
import numpy as np
import pytest
from morie.fn.almteb import alammar_mteb_benchmark_score


def test_almteb_basic():
    """Test basic functionality."""
    task_scores = np.random.default_rng(42).normal(0, 1, 100)
    category_map = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mteb_benchmark_score(task_scores, category_map)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almteb_edge():
    """Test edge cases."""
    task_scores = np.random.default_rng(42).normal(0, 1, 100)
    category_map = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_mteb_benchmark_score(task_scores, category_map)
    assert isinstance(result, dict)
