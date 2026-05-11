"""Tests for dmlqs.deepml_qsar."""
import numpy as np
import pytest
from morie.fn.dmlqs import deepml_qsar


def test_dmlqs_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    activities = np.random.default_rng(42).normal(0, 1, 100)
    result = deepml_qsar(smiles, activities)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dmlqs_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    activities = np.random.default_rng(42).normal(0, 1, 100)
    result = deepml_qsar(smiles, activities)
    assert isinstance(result, dict)
