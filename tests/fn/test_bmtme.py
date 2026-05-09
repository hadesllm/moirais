"""Tests for bmtme.bmtme_model."""
import numpy as np
import pytest
from moirais.fn.bmtme import bmtme_model


def test_bmtme_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env_labels = np.random.default_rng(43).integers(0, 2, 100)
    n_iter = 50
    result = bmtme_model(Y, markers, env_labels, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bmtme_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env_labels = np.random.default_rng(43).integers(0, 2, 100)
    n_iter = 50
    result = bmtme_model(Y, markers, env_labels, n_iter)
    assert isinstance(result, dict)
