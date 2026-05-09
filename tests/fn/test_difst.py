"""Tests for difst -- Stocking-Lord DIF."""
from moirais.fn.difst import dif_stocking_lord
from moirais.fn._containers import DIFResult


class TestDifStockingLord:
    def test_identical_params(self):
        ref = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        foc = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        result = dif_stocking_lord(ref, foc)
        assert isinstance(result, DIFResult)
        assert len(result.flagged) == 0

    def test_different_params(self):
        ref = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        foc = {f"i{j}": {"a": 2.0, "b": 2.0} for j in range(4)}
        result = dif_stocking_lord(ref, foc, threshold=0.01)
        assert len(result.flagged) > 0
