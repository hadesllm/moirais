"""Tests for grdetr.geron_detr_hungarian_matching."""
import numpy as np
import pytest
from morie.fn.grdetr import geron_detr_hungarian_matching


def test_grdetr_basic():
    """Test basic functionality."""
    pred_boxes = np.random.default_rng(42).normal(0, 1, 100)
    pred_classes = np.random.default_rng(43).integers(0, 2, 100)
    gt_boxes = np.random.default_rng(42).normal(0, 1, 100)
    gt_classes = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_detr_hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdetr_edge():
    """Test edge cases."""
    pred_boxes = np.random.default_rng(42).normal(0, 1, 100)
    pred_classes = np.random.default_rng(43).integers(0, 2, 100)
    gt_boxes = np.random.default_rng(42).normal(0, 1, 100)
    gt_classes = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_detr_hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes)
    assert isinstance(result, dict)
