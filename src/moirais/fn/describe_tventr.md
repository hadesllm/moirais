# tventr - Total variation distance

## WHAT IT DOES

Symmetric distance between two discrete distributions: half the L1
distance between their probability vectors. Range [0, 1]; 0 means
identical, 1 means disjoint supports.

## WHEN TO USE

- You need a symmetric, bounded distance between distributions.
- Comparing discrete distributions where KL is undefined or infinite.
- Quantifying how much a distribution can drift before being
  practically distinguishable.

## WHEN NOT TO USE

- You need information-theoretic semantics - use KL.
- Continuous distributions with disjoint support - Wasserstein is
  more useful.

## ASSUMPTIONS

- P and Q sum to 1 (auto-normalized).
- Same support.

## FORMULA

```
TV(P, Q) = (1/2) sum_i |p_i - q_i|
```

## INPUTS / OUTPUTS

```
tventr(p, q) -> RichResult
  p, q    non-negative weights
  returns TV distance, range, support size.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import tventr
>>> tventr([0.5, 0.5], [0.2, 0.8]).statistic
0.3
```

## COMMON MISTAKES

- Treating TV as the L1 distance directly - the 1/2 factor matters.
- Comparing TV across distributions with different supports.

## REFERENCES

- Cover & Thomas (2006).
