"""Test transfer_function (trfnc)."""
import numpy as np
from morie.fn.trfnc import transfer_function, trfnc
from morie.fn._containers import DescriptiveResult


class TestTransferFunction:
    def test_basic(self):
        b = [1.0, 0.5]
        a = [1.0, -0.8]
        result = transfer_function(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "transfer_function"

    def test_magnitude(self):
        b = [1.0]
        a = [1.0]
        result = transfer_function(b, a)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert trfnc is transfer_function
