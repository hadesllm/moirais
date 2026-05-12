# morie.fn -- function file (hadesllm/morie)
"""Hash table with chaining. 'This is the way.' -- Din Djarin"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hash_table(
    keys: list,
    values: list,
    size: int | None = None,
) -> DescriptiveResult:
    """
    Build a hash table with separate chaining for collision resolution.

    :param keys: List of keys (strings or integers).
    :param values: List of corresponding values.
    :param size: Table size. Defaults to 2 * len(keys).
    :return: DescriptiveResult with load factor, collision stats, and table.
    :raises ValueError: If keys and values differ in length.

    References
    ----------
    Knuth, D. E. (1998). *The Art of Computer Programming*, Vol. 3:
    Sorting and Searching. 2nd ed. Addison-Wesley. Sec. 6.4.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have equal length.")
    if not keys:
        raise ValueError("keys must be non-empty.")

    if size is None:
        size = max(2 * len(keys), 1)
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}.")

    table: list[list[tuple]] = [[] for _ in range(size)]

    for k, v in zip(keys, values):
        h = hash(k) % size
        replaced = False
        for idx, (ek, _) in enumerate(table[h]):
            if ek == k:
                table[h][idx] = (k, v)
                replaced = True
                break
        if not replaced:
            table[h].append((k, v))

    chain_lengths = [len(bucket) for bucket in table]
    n_collisions = sum(max(0, cl - 1) for cl in chain_lengths)
    n_empty = sum(1 for cl in chain_lengths if cl == 0)
    load_factor = len(keys) / size

    return DescriptiveResult(
        name="Hash Table",
        value=float(load_factor),
        extra={
            "table_size": size,
            "n_entries": len(keys),
            "load_factor": float(load_factor),
            "n_collisions": n_collisions,
            "n_empty_slots": n_empty,
            "max_chain_length": int(max(chain_lengths)),
            "mean_chain_length": float(np.mean(chain_lengths)),
        },
    )


short = hash_table


def cheatsheet() -> str:
    return "hash_table({}) -> Hash table with chaining. 'This is the way.' -- Din Djarin"
