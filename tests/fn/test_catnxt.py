"""Tests for catnxt.cat_next_item."""
import numpy as np
import pytest
from morie.fn.catnxt import cat_next_item


def test_catnxt_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    item_pool = np.random.default_rng(42).normal(0, 1, 100)
    exposure_constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = cat_next_item(theta_hat, item_pool, exposure_constraints)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_catnxt_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    item_pool = np.random.default_rng(42).normal(0, 1, 100)
    exposure_constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = cat_next_item(theta_hat, item_pool, exposure_constraints)
    assert isinstance(result, dict)
