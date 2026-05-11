"""Tests for spcar.schabenberger_car_model."""
import numpy as np
import pytest
from morie.fn.spcar import schabenberger_car_model


def test_spcar_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_car_model(z, w, covariates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcar_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_car_model(z, w, covariates)
    assert isinstance(result, dict)
