"""Tests for medSEM.sem_mediation."""
import numpy as np
import pytest
from moirais.fn.medSEM import sem_mediation


def test_medSEM_basic():
    """Test basic functionality."""
    model_spec = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_mediation(model_spec, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medSEM_edge():
    """Test edge cases."""
    model_spec = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_mediation(model_spec, data)
    assert isinstance(result, dict)
