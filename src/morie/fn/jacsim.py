# morie.fn — function file (hadesllm/morie)
"""Jaccard similarity."""

def jacsim(a, b) -> float:
    """Jaccard similarity for two sets: |A ∩ B| / |A ∪ B|."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
