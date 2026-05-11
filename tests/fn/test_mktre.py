"""Test merkle_tree."""
from morie.fn._containers import DescriptiveResult
from morie.fn.mktre import merkle_tree, mktre


class TestMerkleTree:
    def test_basic(self):
        result = merkle_tree([b"a", b"b", b"c", b"d"])
        assert isinstance(result, DescriptiveResult)
        assert result.value == 4.0

    def test_root_and_leaf_count(self):
        result = merkle_tree([b"a", b"b", b"c", b"d"])
        assert result.extra["leaf_count"] == 4
        assert result.extra["root"] is not None

    def test_deterministic(self):
        r1 = merkle_tree([b"a", b"b", b"c", b"d"])
        r2 = merkle_tree([b"a", b"b", b"c", b"d"])
        assert r1.extra["root"] == r2.extra["root"]

    def test_alias(self):
        assert mktre is merkle_tree
