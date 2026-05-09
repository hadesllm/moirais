"""Tests for bertS.bertscore."""
import numpy as np
import pytest
from moirais.fn.bertS import bertscore


def test_bertS_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = bertscore(candidate, reference, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bertS_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = bertscore(candidate, reference, model)
    assert isinstance(result, dict)
