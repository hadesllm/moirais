"""Test step_response (stprs)."""
import numpy as np
from morie.fn.stprs import step_response, stprs
from morie.fn._containers import SignalResult


class TestStepResponse:
    def test_basic(self):
        b = [1.0]
        a = [1.0]
        result = step_response(b, a, N=50)
        assert isinstance(result, SignalResult)
        assert result.name == "step_response"

    def test_unity(self):
        b = [1.0]
        a = [1.0]
        result = step_response(b, a, N=10)
        np.testing.assert_allclose(result.filtered, np.ones(10))

    def test_alias(self):
        assert stprs is step_response
