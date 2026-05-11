"""Tests for morie.fn.magnm -- Biot-Savart magnetic field."""

import numpy as np
from morie.fn.magnm import biot_savart, magnm
from morie.fn._containers import DescriptiveResult


class TestMagnm:
    def test_alias(self):
        assert magnm is biot_savart

    def test_straight_wire(self):
        pts = np.array([[0, 0, -1], [0, 0, 1]], dtype=float)
        fp = np.array([0.1, 0, 0])
        r = biot_savart(pts, current=1.0, field_point=fp)
        assert isinstance(r, DescriptiveResult)
        assert r.value["B_mag"] > 0

    def test_zero_current(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        fp = np.array([0.5, 1, 0])
        r = biot_savart(pts, current=0.0, field_point=fp)
        assert r.value["B_mag"] < 1e-15
