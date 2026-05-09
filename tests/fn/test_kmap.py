"""Tests for kmap.kamath_autoprompt_gradient_search."""
import numpy as np
import pytest
from moirais.fn.kmap import kamath_autoprompt_gradient_search


def test_kmap_basic():
    """Test basic functionality."""
    template = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmap_edge():
    """Test edge cases."""
    template = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)
