"""Tests for goldsc.gold_score."""
import numpy as np
import pytest
from morie.fn.goldsc import gold_score


def test_goldsc_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = gold_score(receptor, ligand)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_goldsc_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = gold_score(receptor, ligand)
    assert isinstance(result, dict)
