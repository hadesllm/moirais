"""Tests for morie.fn.swaopt -- particle swarm optimization."""

import numpy as np
from morie.fn.swaopt import swarm_optimize, swaopt
from morie.fn._containers import DescriptiveResult


class TestSwaopt:
    def test_alias(self):
        assert swaopt is swarm_optimize

    def test_sphere(self):
        def sphere(x):
            return float(np.sum(x ** 2))
        r = swarm_optimize(sphere, [(-5, 5), (-5, 5)], n_particles=20, n_iter=50, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert r.value["f_best"] < 1.0

    def test_convergence(self):
        def f(x):
            return float((x[0] - 3) ** 2)
        r = swarm_optimize(f, [(0, 10)], seed=0)
        assert len(r.value["convergence"]) > 1
