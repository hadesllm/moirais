"""Tests for moirais.fn.wkndm -- multi-objective Pareto optimization."""

import numpy as np
from moirais.fn.wkndm import pareto_optimize, wkndm
from moirais.fn._containers import DescriptiveResult


class TestWkndm:
    def test_alias(self):
        assert wkndm is pareto_optimize

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
