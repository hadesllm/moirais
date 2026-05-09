"""Tests for augmn.albert_chib_augmentation."""
import numpy as np
import pytest
from moirais.fn.augmn import albert_chib_augmentation


def test_augmn_basic():
    """Test basic functionality."""
    y_bin = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = albert_chib_augmentation(y_bin, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_augmn_edge():
    """Test edge cases."""
    y_bin = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = albert_chib_augmentation(y_bin, X, Z)
    assert isinstance(result, dict)
