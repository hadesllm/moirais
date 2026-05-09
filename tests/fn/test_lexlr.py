"""Tests for moirais.fn.lexlr -- Lexicographic rank aggregation."""

import numpy as np
from moirais.fn.lexlr import lexico_rank, lexlr
from moirais.fn._containers import DescriptiveResult


class TestLexlr:
    def test_alias(self):
        assert lexlr is lexico_rank

    def test_unanimous(self):
        rankings = np.array([[1, 2, 3], [1, 2, 3]])
        result = lexico_rank(rankings)
        assert isinstance(result, DescriptiveResult)
        assert result.value[0] == 0

    def test_three_items(self):
        rankings = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]])
        result = lexico_rank(rankings)
        assert len(result.value) == 3
