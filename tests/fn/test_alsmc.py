"""Tests for alsmc.alammar_simcse_dropout_aug."""
import numpy as np
import pytest
from morie.fn.alsmc import alammar_simcse_dropout_aug


def test_alsmc_basic():
    """Test basic functionality."""
    embeddings_dropout1 = np.random.default_rng(42).normal(0, 1, 100)
    embeddings_dropout2 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_simcse_dropout_aug(embeddings_dropout1, embeddings_dropout2, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alsmc_edge():
    """Test edge cases."""
    embeddings_dropout1 = np.random.default_rng(42).normal(0, 1, 100)
    embeddings_dropout2 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_simcse_dropout_aug(embeddings_dropout1, embeddings_dropout2, tau)
    assert isinstance(result, dict)
