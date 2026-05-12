# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Our greatest glory is not in never falling, but in rising every time we fall. -- Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def binary_heap(
    data: np.ndarray,
    heap_type: str = "min",
) -> DescriptiveResult:
    """
    Build a binary heap and return the sorted extraction order.

    Implements heapify-up insertion and extract-root to produce a
    heap-sorted sequence.

    :param data: Array of numeric values.
    :param heap_type: "min" or "max". Default "min".
    :return: DescriptiveResult with sorted output and heap array.
    :raises ValueError: If heap_type is not "min" or "max".

    References
    ----------
    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
    (2009). *Introduction to Algorithms*. 3rd ed. MIT Press. Ch. 6.
    """
    if heap_type not in ("min", "max"):
        raise ValueError(f"heap_type must be 'min' or 'max', got '{heap_type}'.")

    arr = np.asarray(data, dtype=np.float64).ravel().tolist()
    n = len(arr)
    is_min = heap_type == "min"

    heap: list[float] = []

    def _sift_up(h: list[float], idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2
            if (is_min and h[idx] < h[parent]) or (not is_min and h[idx] > h[parent]):
                h[idx], h[parent] = h[parent], h[idx]
                idx = parent
            else:
                break

    def _sift_down(h: list[float], idx: int, size: int) -> None:
        while True:
            smallest = idx
            left, right = 2 * idx + 1, 2 * idx + 2
            if left < size:
                if (is_min and h[left] < h[smallest]) or (not is_min and h[left] > h[smallest]):
                    smallest = left
            if right < size:
                if (is_min and h[right] < h[smallest]) or (not is_min and h[right] > h[smallest]):
                    smallest = right
            if smallest != idx:
                h[idx], h[smallest] = h[smallest], h[idx]
                idx = smallest
            else:
                break

    for val in arr:
        heap.append(val)
        _sift_up(heap, len(heap) - 1)

    heap_snapshot = list(heap)
    sorted_output = []
    size = len(heap)
    for _ in range(size):
        sorted_output.append(heap[0])
        heap[0] = heap[size - 1]
        size -= 1
        if size > 0:
            _sift_down(heap, 0, size)

    return DescriptiveResult(
        name=f"{'Min' if is_min else 'Max'} Binary Heap",
        value=float(sorted_output[0]) if sorted_output else float("nan"),
        extra={
            "sorted": sorted_output,
            "heap_array": heap_snapshot,
            "heap_type": heap_type,
            "n": n,
        },
    )


short = binary_heap


def cheatsheet() -> str:
    return "binary_heap({}) -> Binary heap operations. 'You were the chosen one!' -- Obi-Wa"
