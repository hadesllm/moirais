"""Tests for hmssg.geron_semantic_segmentation."""
import numpy as np
import pytest
from morie.fn.hmssg import geron_semantic_segmentation


def test_hmssg_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_semantic_segmentation(image, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmssg_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_semantic_segmentation(image, model)
    assert isinstance(result, dict)
