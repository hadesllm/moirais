"""Tests for moirais.fn.dvstr -- ensemble stacking."""

import numpy as np
from moirais.fn.dvstr import ensemble_stack, dvstr
from moirais.fn._containers import DescriptiveResult


class TestDvstr:
    def test_alias(self):
        assert dvstr is ensemble_stack

    def test_basic_stack(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        preds = np.column_stack([y + rng.normal(0, 0.5, 100),
                                  y + rng.normal(0, 0.3, 100)])
        r = ensemble_stack(preds, y)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["cv_rmse"] < 1.0

    def test_weights_exist(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 50)
        preds = np.column_stack([y, y * 0.5])
        r = ensemble_stack(preds, y)
        assert len(r.extra["weights"]) == 2
