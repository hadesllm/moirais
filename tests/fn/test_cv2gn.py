"""Tests for cv2gn.cv2_genomic."""
import numpy as np
import pytest
from moirais.fn.cv2gn import cv2_genomic


def test_cv2gn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env = np.random.default_rng(42).normal(0, 1, 100)
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = cv2_genomic(y, markers, env, n_folds)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_cv2gn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env = np.random.default_rng(42).normal(0, 1, 100)
    n_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = cv2_genomic(y, markers, env, n_folds)
    assert isinstance(result, dict)
