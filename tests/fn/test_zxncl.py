"""Tests for morie.fn.zxncl -- Network spatial clustering"""

import numpy as np
import pytest

from morie.fn.zxncl import network_cluster_sp


class TestNetworkClusterSp:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = network_cluster_sp(observed)
        assert result.value is not None

    def test_output_type(self):
        result = network_cluster_sp(np.array([1,2,3,4,5]))
        assert hasattr(result, "value")
