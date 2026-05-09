"""Tests for moirais.fn.blade -- Canny edge detection."""

import numpy as np
from moirais.fn.blade import edge_detect, blade
from moirais.fn._containers import DescriptiveResult


class TestBlade:
    def test_alias(self):
        assert blade is edge_detect

    def test_detects_edges(self):
        img = np.zeros((50, 50))
        img[10:40, 10:40] = 1.0
        r = edge_detect(img, sigma=1.0)
        assert isinstance(r, DescriptiveResult)
        assert r.value["n_edge_pixels"] > 0

    def test_gradient_shape(self):
        img = np.zeros((30, 30))
        img[:15, :] = 1.0
        r = edge_detect(img, sigma=1.0)
        assert r.value["gradient_magnitude"].shape == (30, 30)
