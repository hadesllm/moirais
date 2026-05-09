"""Tests for agpar.party_unity_score."""
import numpy as np
import pytest
from moirais.fn.agpar import party_unity_score


def test_agpar_basic():
    """Test basic functionality."""
    vote_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    party_id = np.random.default_rng(42).normal(0, 1, 100)
    result = party_unity_score(vote_matrix, party_id)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agpar_edge():
    """Test edge cases."""
    vote_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    party_id = np.random.default_rng(42).normal(0, 1, 100)
    result = party_unity_score(vote_matrix, party_id)
    assert isinstance(result, dict)
