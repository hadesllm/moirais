"""morie.fast — the opt-in JIT acceleration surface.

Public re-export of the kernels in :mod:`morie._jit`.  ``import morie.fast``
gives a stable, user-facing namespace whose functions are
numerically identical to their scipy / numpy equivalents but compile
to native code when ``numba`` is installed.

Usage
-----

    >>> import morie.fast as mf
    >>> mf.is_jit_available()
    True            # after `pip install morie[fast]`
    False           # otherwise (kernels still work, just no JIT)
    >>> x = np.random.normal(size=10_000)
    >>> mf.normal_pdf(x, 0.0, 1.0)
    array([...])

Functions
---------

``normal_pdf(x, mean, sd)``
    Drop-in for ``scipy.stats.norm.pdf``.  ~5-10× faster on small
    arrays once JIT-compiled.

``normal_logpdf(x, mean, sd)``
    Drop-in for ``scipy.stats.norm.logpdf``.

``mean_jit(arr)`` / ``std_jit(arr, ddof=1)`` / ``var_jit(arr, ddof=1)``
    Drop-ins for ``arr.mean()`` / ``np.std(arr, ddof=...)`` /
    ``np.var(arr, ddof=...)``.

``cor_pearson_jit(x, y)``
    Drop-in for ``scipy.stats.pearsonr(x, y)[0]`` (point estimate only).

``euclid_dist_jit(a, b)``
    Drop-in for ``np.linalg.norm(a - b)``.

``is_jit_available()``
    Returns ``True`` iff numba is importable and ``MORIE_DISABLE_JIT``
    is not set.

``jit_if_available(*args, **kwargs)``
    Decorator factory that passes through to ``numba.njit`` when
    numba is installed, otherwise no-ops.  For users decorating their
    own kernels::

        from morie.fast import jit_if_available

        @jit_if_available(cache=True, nogil=True)
        def my_hot_loop(arr):
            ...

When numba is NOT installed (e.g. Python 3.15+, where numba wheels are
not yet available), every function still runs as plain numpy.
Numerically identical, just slower.  ``is_jit_available()`` lets the
caller branch on this if perf-sensitive reporting matters.
"""

from __future__ import annotations

from ._jit import (
    cor_pearson_jit,
    euclid_dist_jit,
    is_jit_available,
    mean_jit,
    normal_logpdf,
    normal_pdf,
    std_jit,
    var_jit,
)


def jit_if_available(*args, **kwargs):
    """Conditional ``numba.njit`` decorator.

    Returns ``numba.njit(*args, **kwargs)`` when numba is importable
    and ``MORIE_DISABLE_JIT`` is unset; returns an identity decorator
    otherwise.

    The signature mirrors ``numba.njit`` so existing JIT-decorated
    code can swap to this drop-in without other changes.

    Use either form::

        @jit_if_available
        def f(x): ...

        @jit_if_available(cache=True, nogil=True)
        def f(x): ...
    """
    if not is_jit_available():
        # Identity-decorator forms — handle both bare and call-style:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _passthrough(fn):
            return fn
        return _passthrough

    # numba is available; defer to it
    import numba  # noqa: WPS433 — local on purpose; never eager-imported
    return numba.njit(*args, **kwargs)


__all__ = [
    "normal_pdf",
    "normal_logpdf",
    "mean_jit",
    "std_jit",
    "var_jit",
    "cor_pearson_jit",
    "euclid_dist_jit",
    "is_jit_available",
    "jit_if_available",
]
