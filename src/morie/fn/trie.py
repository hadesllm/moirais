"""Prefix trie operations. 'Hope is like the sun.' -- Poe Dameron"""

from __future__ import annotations

from ._containers import DescriptiveResult


def trie_operations(
    words: list[str],
    prefix_query: str | None = None,
) -> DescriptiveResult:
    """
    Build a prefix trie and optionally search by prefix.

    :param words: List of words to insert.
    :param prefix_query: Optional prefix to search for matching words.
    :return: DescriptiveResult with trie stats and prefix matches.
    :raises ValueError: If words list is empty.

    References
    ----------
    Fredkin, E. (1960). Trie memory. *Communications of the ACM*,
    3(9), 490-499.
    """
    if not words:
        raise ValueError("Words list must be non-empty.")

    root: dict = {}
    for word in words:
        node = root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node["$"] = True

    def _count_nodes(node: dict) -> int:
        count = 1
        for k, v in node.items():
            if k != "$" and isinstance(v, dict):
                count += _count_nodes(v)
        return count

    n_nodes = _count_nodes(root)

    matches: list[str] = []
    if prefix_query is not None:
        node = root
        for ch in prefix_query:
            if ch not in node:
                node = None
                break
            node = node[ch]

        if node is not None:

            def _collect(n: dict, prefix: str) -> None:
                if "$" in n:
                    matches.append(prefix)
                for k, v in sorted(n.items()):
                    if k != "$" and isinstance(v, dict):
                        _collect(v, prefix + k)

            _collect(node, prefix_query)

    return DescriptiveResult(
        name="Prefix Trie",
        value=float(n_nodes),
        extra={
            "n_words": len(words),
            "n_nodes": n_nodes,
            "n_unique_words": len(set(words)),
            "prefix_query": prefix_query,
            "prefix_matches": matches,
            "n_matches": len(matches),
        },
    )


short = trie_operations


def cheatsheet() -> str:
    return "trie_operations({}) -> Prefix trie operations. 'Hope is like the sun.' -- Poe Damer"
