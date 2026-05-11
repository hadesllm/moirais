"""Tests for alcont.alammar_continued_pretraining_mlm."""
import numpy as np
import pytest
from morie.fn.alcont import alammar_continued_pretraining_mlm


def test_alcont_basic():
    """Test basic functionality."""
    domain_corpus = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    n_mlm_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_continued_pretraining_mlm(domain_corpus, encoder, n_mlm_steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alcont_edge():
    """Test edge cases."""
    domain_corpus = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    n_mlm_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_continued_pretraining_mlm(domain_corpus, encoder, n_mlm_steps)
    assert isinstance(result, dict)
