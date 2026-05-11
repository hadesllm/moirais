"""Torus link: linking number computation."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def torus_link(p: int = 2, q: int = 3) -> DescriptiveResult:
    """Compute the linking number of a (p, q)-torus link.

    For a (p, q)-torus link with gcd(p, q) = d > 1, the link has *d*
    components and the pairwise linking number is p*q / d^2.
    For gcd(p, q) = 1 (torus knot, single component), linking number is 0.

    :param p: First winding number (positive integer).
    :param q: Second winding number (positive integer).
    :return: DescriptiveResult with linking_number and n_components in *extra*.
    """
    if p <= 0 or q <= 0:
        raise ValueError(f"p and q must be positive, got p={p}, q={q}.")
    d = math.gcd(p, q)
    n_components = d
    linking_number = (p * q) // (d * d) if d > 1 else 0
    return DescriptiveResult(
        name="torus_link",
        value=float(linking_number),
        extra={
            "linking_number": linking_number,
            "n_components": n_components,
            "p": p,
            "q": q,
            "gcd": d,
        },
    )


def cheatsheet() -> str:
    return "torus_link(p, q) -> linking number of (p,q)-torus link"
