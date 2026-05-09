"""Tests for kmtoxg.kamath_toxigen_score."""
import numpy as np
import pytest
from moirais.fn.kmtoxg import kamath_toxigen_score


def test_kmtoxg_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_toxigen_score(text, classifier)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmtoxg_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_toxigen_score(text, classifier)
    assert isinstance(result, dict)
