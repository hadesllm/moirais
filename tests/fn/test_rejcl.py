"""Test reject_option (rejcl)."""
import numpy as np
from moirais.fn.rejcl import reject_option, rejcl
from moirais.fn._containers import DescriptiveResult


class TestRejcl:
    def test_basic(self):
        scores = np.array([0.1, 0.45, 0.55, 0.9])
        result = reject_option(scores, threshold=0.5, reject_width=0.1)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "reject_option"
        preds = result.extra["predictions"]
        assert preds[1] == -1
        assert preds[2] == -1

    def test_no_rejection(self):
        scores = np.array([0.0, 1.0])
        result = reject_option(scores, threshold=0.5, reject_width=0.01)
        assert result.extra["n_rejected"] == 0

    def test_alias(self):
        assert rejcl is reject_option
