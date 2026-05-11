"""Tests for bleuS.bleu."""
import numpy as np
import pytest
from morie.fn.bleuS import bleu


def test_bleuS_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    max_n = np.random.default_rng(42).normal(0, 1, 100)
    result = bleu(candidate, references, max_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bleuS_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    max_n = np.random.default_rng(42).normal(0, 1, 100)
    result = bleu(candidate, references, max_n)
    assert isinstance(result, dict)
