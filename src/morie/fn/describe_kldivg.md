# kldivg - Kullback-Leibler divergence (discrete)

## WHAT IT DOES

Reports the relative entropy from distribution Q to P:
sum_i p_i log(p_i/q_i). It quantifies the inefficiency of using Q
to encode messages from P. Asymmetric: KL(P||Q) != KL(Q||P).

## WHEN TO USE

- Comparing how a model distribution Q matches the data distribution P.
- Information-theoretic measures of model fit (e.g., cross-entropy
  loss is essentially KL up to a constant).
- Variational inference (minimize KL between approximating and true).

## WHEN NOT TO USE

- You need a symmetric distance - use total variation (`tventr`) or
  Wasserstein (`wasdst`).
- Q has zero where P > 0 - KL is infinite. Smooth or condition first.

## ASSUMPTIONS

- P and Q are both probability distributions over the same support.
- Discrete; for continuous distributions use a kernel/histogram
  estimator first.

## FORMULA

```
KL(P || Q) = sum_i p_i log(p_i / q_i)
```

## INPUTS / OUTPUTS

```
kldivg(p, q, base=2.0) -> RichResult
  p, q     non-negative weights (auto-normalized)
  base     log base; default 2 (bits)
  returns  divergence, support size, P entropy, Q entropy.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import kldivg
>>> kldivg([0.5, 0.5], [0.4, 0.6]).statistic
0.029  # ish
```

## COMMON MISTAKES

- Using KL as a distance - it's not symmetric and doesn't satisfy the
  triangle inequality.
- Failing to handle zero probabilities in Q.

## REFERENCES

- Cover & Thomas (2006) Elements of Information Theory.
