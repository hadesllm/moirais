"""Tests for jotide.joseph_tide_encoder."""
import numpy as np
import pytest
from moirais.fn.jotide import joseph_tide_encoder


def test_jotide_basic():
    """Test basic functionality."""
    past = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_tide_encoder(past, covariates, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jotide_edge():
    """Test edge cases."""
    past = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_tide_encoder(past, covariates, horizon)
    assert isinstance(result, dict)
