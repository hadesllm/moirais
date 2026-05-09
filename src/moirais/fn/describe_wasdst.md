# wasdst - 1D Wasserstein-1 distance (earth mover's)

## WHAT IT DOES

Reports the cost of transforming one empirical distribution into
another, where cost is total mass-times-distance. Symmetric and
sensitive to scale shifts even when supports don't overlap.

## WHEN TO USE

- Comparing distributions with possibly disjoint supports.
- Distribution drift detection.
- More robust alternative to KL when supports differ.

## WHEN NOT TO USE

- High-dimensional - 1D Wasserstein doesn't capture multivariate
  structure; use entropic-regularized OT in 2D+.
- Speed matters - exact OT is O(n log n) for 1D but expensive in 2D+.

## ASSUMPTIONS

- Numeric samples (1D).
- Independence within each sample.

## FORMULA

```
W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx
```
For empirical distributions, sort both and compute sum of |x_(i) - y_(i)|.

## INPUTS / OUTPUTS

```
wasdst(u, v) -> RichResult
  u, v     1D numeric samples
  returns  W_1 distance, sample sizes, means, mean diff.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import wasdst
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> wasdst(rng.standard_normal(100), rng.standard_normal(100) + 1).statistic
1.0  # ish - they differ by mean shift of 1
```

## COMMON MISTAKES

- Comparing W_1 across different scales (W_1(P, Q) is in the units
  of x).
- Using on multidim data without thinking about which axis to project on.

## REFERENCES

- Villani (2008) Optimal Transport. Wasserman (2004) ch.2.
