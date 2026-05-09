# moirais.fn — function file (hadesllm/moirais)
"""Merkle tree construction and verification."""

from __future__ import annotations

from ._containers import DescriptiveResult


def merkle_tree(leaves: list[bytes], hash_fn: str = "sha3_256") -> DescriptiveResult:
    """Build a Merkle tree from leaf values.

    :param leaves: List of leaf byte values.
    :param hash_fn: Hash function name (currently sha3_256 only).
    :return: DescriptiveResult with root hash, tree structure, leaf count.
    """
    from moirais.crypto._hashsig import merkle_auth_path, merkle_tree_build, merkle_verify

    result = merkle_tree_build(leaves)
    return DescriptiveResult(
        name="merkle_tree",
        value=float(result["leaf_count"]),
        extra={
            "root": result["root"],
            "tree": result["tree"],
            "leaf_count": result["leaf_count"],
            "height": len(result["tree"]) - 1 if result["tree"] else 0,
            "verify_fn": merkle_verify,
            "auth_path_fn": merkle_auth_path,
        },
    )


mktre = merkle_tree


def cheatsheet() -> str:
    return "merkle_tree({}) -> Merkle tree construction and verification."
