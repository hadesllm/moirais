"""Tests for masrcn.mask_rcnn_segmentation."""
import numpy as np
import pytest
from moirais.fn.masrcn import mask_rcnn_segmentation


def test_masrcn_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    proposals = np.random.default_rng(42).normal(0, 1, 100)
    result = mask_rcnn_segmentation(image, proposals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_masrcn_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    proposals = np.random.default_rng(42).normal(0, 1, 100)
    result = mask_rcnn_segmentation(image, proposals)
    assert isinstance(result, dict)
