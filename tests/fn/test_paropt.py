"""Tests for morie.fn.paropt -- multi-objective Pareto optimization."""

import numpy as np
from morie.fn.paropt import pareto_optimize, paropt
from morie.fn._containers import DescriptiveResult


class TestParopt:
    def test_alias(self):
        assert paropt is pareto_optimize

    def test_pareto_front(self):
        rng = np.random.default_rng(42)
        O = rng.uniform(0, 1, (20, 2))
        r = pareto_optimize(O)
        assert isinstance(r, DescriptiveResult)
        assert r.value["n_pareto"] > 0
        assert r.value["n_pareto"] <= 20

    def test_all_pareto(self):
        O = np.array([[1, 2], [2, 1]])
        r = pareto_optimize(O)
        assert r.value["n_pareto"] == 2
