"""Tests for morie.fn.augla -- Augmented Lagrangian."""

import numpy as np
from morie.fn.augla import augmented_lagrangian, augla
from morie.fn._containers import DescriptiveResult


class TestAugla:
    def test_alias(self):
        assert augla is augmented_lagrangian

    def test_constrained_min(self):
        f = lambda x: x[0]**2 + x[1]**2
        g = lambda x: np.array([2 * x[0], 2 * x[1]])
        cons = [lambda x: x[0] + x[1] - 1]
        r = augmented_lagrangian(f, g, cons, np.array([2.0, 2.0]))
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["x"][0] + r.extra["x"][1] - 1.0) < 0.1
        assert abs(r.extra["x"][0] - 0.5) < 0.2
