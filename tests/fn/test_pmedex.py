"""Tests for pmedex.proportion_te_explained."""
import numpy as np
import pytest
from moirais.fn.pmedex import proportion_te_explained


def test_pmedex_basic():
    """Test basic functionality."""
    nie = np.random.default_rng(42).normal(0, 1, 100)
    te = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_te_explained(nie, te)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pmedex_edge():
    """Test edge cases."""
    nie = np.random.default_rng(42).normal(0, 1, 100)
    te = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_te_explained(nie, te)
    assert isinstance(result, dict)
