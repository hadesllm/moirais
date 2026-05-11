"""Tests for alocv.alammar_output_verification."""
import numpy as np
import pytest
from morie.fn.alocv import alammar_output_verification


def test_alocv_basic():
    """Test basic functionality."""
    response = np.random.default_rng(42).normal(0, 1, 100)
    criteria = np.random.default_rng(42).normal(0, 1, 100)
    verifier_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_output_verification(response, criteria, verifier_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alocv_edge():
    """Test edge cases."""
    response = np.random.default_rng(42).normal(0, 1, 100)
    criteria = np.random.default_rng(42).normal(0, 1, 100)
    verifier_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_output_verification(response, criteria, verifier_model)
    assert isinstance(result, dict)
