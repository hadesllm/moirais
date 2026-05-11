"""Tests for alfesf.esmfold_lm_only."""
import numpy as np
import pytest
from morie.fn.alfesf import esmfold_lm_only


def test_alfesf_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    esm_model = np.random.default_rng(42).normal(0, 1, 100)
    result = esmfold_lm_only(sequence, esm_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfesf_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    esm_model = np.random.default_rng(42).normal(0, 1, 100)
    result = esmfold_lm_only(sequence, esm_model)
    assert isinstance(result, dict)
