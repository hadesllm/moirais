"""Tests for morie.fn.ucr -- UCR offense classification."""

from morie.fn.ucr import ucr_classify, ucr
from morie.fn._containers import DescriptiveResult


class TestUcr:
    def test_alias(self):
        assert ucr is ucr_classify

    def test_classify_known(self):
        result = ucr_classify(["murder", "theft", "assault"])
        assert isinstance(result, DescriptiveResult)
        df = result.value
        assert len(df) == 3
        assert all(df["part"].isin(["Part I", "Part II"]))

    def test_unknown_offense(self):
        result = ucr_classify(["jaywalking"])
        df = result.value
        assert df.iloc[0]["part"] == "Unclassified"
