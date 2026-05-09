"""Tests for nwcen -- network centrality."""
import numpy as np
from moirais.fn.nwcen import network_centrality
from moirais.fn._containers import DescriptiveResult


class TestNetworkCentrality:
    def test_basic(self):
        A = np.array([[0, 0.5, 0.3], [0.5, 0, 0.2], [0.3, 0.2, 0]])
        result = network_centrality(A)
        assert isinstance(result, DescriptiveResult)
        assert "strength" in result.value.columns

    def test_isolated_node(self):
        A = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
        result = network_centrality(A)
        df = result.value
        assert df.loc[2, "strength"] == 0.0
