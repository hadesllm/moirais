"""Tests for hmcst.geron_contrastive_learning."""
import numpy as np
import pytest
from moirais.fn.hmcst import geron_contrastive_learning


def test_hmcst_basic():
    """Test basic functionality."""
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_contrastive_learning(embeddings, positives)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcst_edge():
    """Test edge cases."""
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    positives = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_contrastive_learning(embeddings, positives)
    assert isinstance(result, dict)
