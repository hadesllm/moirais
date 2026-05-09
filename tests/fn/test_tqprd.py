"""Tests for moirais.fn.tqprd — TurboQuant product quantizer."""

import numpy as np
import pytest

from moirais.fn.tqprd import turboquant_prod


class TestTurboquantProd:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = turboquant_prod(x, bits=3, n_sub=2)
        assert res.name == "turboquant_prod"
        assert res.value >= 0

    def test_sub_vector_count(self):
        x = np.random.default_rng(0).standard_normal(128)
        res = turboquant_prod(x, bits=3, n_sub=4)
        assert len(res.extra["codes"]) == 4
        assert len(res.extra["centroids"]) == 4

    def test_compression_ratio(self):
        x = np.random.default_rng(1).standard_normal(64)
        res = turboquant_prod(x, bits=4)
        assert res.extra["compression_ratio"] == pytest.approx(8.0)
