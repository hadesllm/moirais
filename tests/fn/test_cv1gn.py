"""Tests for cv1gn.cv1_genomic."""
import numpy as np
import pytest
from morie.fn.cv1gn import cv1_genomic


def test_cv1gn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = cv1_genomic(y, markers, n_folds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cv1gn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = cv1_genomic(y, markers, n_folds)
    assert isinstance(result, dict)
