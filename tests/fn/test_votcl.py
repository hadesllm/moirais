"""Test voting_classify (votcl)."""
import numpy as np
from morie.fn.votcl import voting_classify, votcl
from morie.fn._containers import DescriptiveResult


class TestVotcl:
    def test_basic(self):
        preds = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1]])
        result = voting_classify(preds)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "voting_classify"
        assert len(result.extra["predictions"]) == 4

    def test_majority(self):
        preds = np.array([[1, 0, 1], [1, 1, 0], [1, 0, 0]])
        result = voting_classify(preds)
        assert result.extra["predictions"][0] == 1

    def test_alias(self):
        assert votcl is voting_classify
