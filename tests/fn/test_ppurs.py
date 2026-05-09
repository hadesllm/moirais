"""Tests for moirais.fn.ppurs -- Projection pursuit."""

import numpy as np
from moirais.fn.ppurs import projection_pursuit, ppurs
from moirais.fn._containers import DescriptiveResult


class TestProjectionPursuit:
    def test_alias(self):
        assert ppurs is projection_pursuit

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        res = projection_pursuit(X, n_components=2)
        assert isinstance(res, DescriptiveResult)

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        res = projection_pursuit(X, n_components=2)
        assert res.value.shape == (50, 2)
        assert len(res.extra["index_values"]) == 2

    def test_negentropy_index(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        res = projection_pursuit(X, n_components=1, index="negentropy")
        assert res.value.shape == (50, 1)
