"""Tests for moirais.fn.zxnsp -- Network shortest path spatial"""

import numpy as np
import pytest

from moirais.fn.zxnsp import network_shortest


class TestNetworkShortest:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = network_shortest(data)
        assert result.value is not None

    def test_output_type(self):
        result = network_shortest(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
