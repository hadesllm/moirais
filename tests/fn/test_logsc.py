"""Test logit_softcap."""
import numpy as np
from morie.fn.logsc import logit_softcap, logsc
from morie.fn._containers import DescriptiveResult


class TestLogitSoftcap:
    def test_basic(self):
        logits = np.array([-100.0, 0.0, 100.0])
        result = logit_softcap(logits, cap=30.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "logit_softcap"

    def test_bounded(self):
        logits = np.array([1000.0, -1000.0])
        result = logit_softcap(logits, cap=30.0)
        out = result.extra["output"]
        assert np.all(np.abs(out) <= 30.0 + 1e-10)

    def test_near_zero(self):
        logits = np.array([0.01])
        result = logit_softcap(logits, cap=30.0)
        assert abs(result.extra["output"][0] - 0.01) < 0.001

    def test_alias(self):
        assert logsc is logit_softcap
