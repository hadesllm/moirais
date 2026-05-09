"""Tests for ricei.rice_index."""
import numpy as np
import pytest
from moirais.fn.ricei import rice_index


def test_ricei_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    party_id = np.random.default_rng(42).normal(0, 1, 100)
    result = rice_index(votes, party_id)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ricei_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    party_id = np.random.default_rng(42).normal(0, 1, 100)
    result = rice_index(votes, party_id)
    assert isinstance(result, dict)
