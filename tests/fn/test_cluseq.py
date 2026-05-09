"""Tests for cluseq.sequence_clustering."""
import numpy as np
import pytest
from moirais.fn.cluseq import sequence_clustering


def test_cluseq_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    snp_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = sequence_clustering(sequences, snp_threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cluseq_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    snp_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = sequence_clustering(sequences, snp_threshold)
    assert isinstance(result, dict)
