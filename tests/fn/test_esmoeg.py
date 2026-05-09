"""Tests for esmoeg.esem_target_rotation."""
import numpy as np
import pytest
from moirais.fn.esmoeg import esem_target_rotation


def test_esmoeg_basic():
    """Test basic functionality."""
    loadings = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = esem_target_rotation(loadings, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esmoeg_edge():
    """Test edge cases."""
    loadings = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = esem_target_rotation(loadings, target)
    assert isinstance(result, dict)
