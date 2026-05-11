"""Tests for morie.fn.mhstg -- stigma index."""

import pytest
from morie.fn.mhstg import stigma_index


class TestStigmaIndex:
    def test_basic(self):
        res = stigma_index([3, 4, 2, 5, 1], max_per_item=5)
        assert res.measure == "stigma_index"
        assert res.estimate == pytest.approx(15 / 25 * 100)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stigma_index([])
