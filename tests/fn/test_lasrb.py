"""Tests for moirais.fn.lasrb -- IoU metric."""

import numpy as np
from moirais.fn.lasrb import iou_metric, lasrb
from moirais.fn._containers import DescriptiveResult


class TestLasrb:
    def test_alias(self):
        assert lasrb is iou_metric

    def test_perfect_overlap(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        r = iou_metric(boxes, boxes)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - 1.0) < 1e-10

    def test_no_overlap(self):
        pred = np.array([[0, 0, 5, 5]], dtype=float)
        true = np.array([[10, 10, 20, 20]], dtype=float)
        r = iou_metric(pred, true)
        assert r.value == 0.0
