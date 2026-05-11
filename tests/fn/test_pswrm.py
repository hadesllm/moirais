"""Tests for morie.fn.pswrm -- Particle swarm optimization."""

import numpy as np
from morie.fn.pswrm import particle_swarm, pswrm
from morie.fn._containers import DescriptiveResult


class TestPswrm:
    def test_alias(self):
        assert pswrm is particle_swarm

    def test_sphere(self):
        f = lambda x: np.sum(x**2)
        r = particle_swarm(f, [(-5, 5), (-5, 5)])
        assert isinstance(r, DescriptiveResult)
        assert r.value < 0.1
        assert np.all(np.abs(r.extra["x"]) < 1.0)

    def test_1d(self):
        f = lambda x: (x[0] - 3)**2
        r = particle_swarm(f, [(0, 10)])
        assert abs(r.extra["x"][0] - 3.0) < 0.5
