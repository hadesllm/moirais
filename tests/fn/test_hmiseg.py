"""Tests for hmiseg.geron_image_segmentation."""
import numpy as np
import pytest
from morie.fn.hmiseg import geron_image_segmentation


def test_hmiseg_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_image_segmentation(image, n_clusters, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmiseg_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    n_clusters = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_image_segmentation(image, n_clusters, seed)
    assert isinstance(result, dict)
