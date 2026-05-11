# npowtt - Required n per group for two-sample t-test power

## WHAT IT DOES

Iteratively finds the smallest equal n per group that achieves a
target power for a two-sample t-test detecting a given effect size d.

## WHEN TO USE

- Sample-size planning at the study-design stage.
- Grant proposals requiring justified n.
- Comparing study costs against required power.

## WHEN NOT TO USE

- Unequal group allocation needed - implement the binary search yourself.
- Adaptive designs - need different methods.

## ASSUMPTIONS

- Equal group sizes.
- Two-sample t with equal variances.
- d is the smallest effect of practical interest.

## FORMULA

Smallest n where powtt2(d, n, n, alpha) >= target_power, by linear search.

## INPUTS / OUTPUTS

```
npowtt(d, target_power=0.8, alpha=0.05, max_n=10000) -> RichResult
  d              effect size to detect
  target_power   conventional default 0.8 (Cohen)
  returns        n per group, total n, achieved power.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import npowtt
>>> npowtt(0.5, 0.8).statistic
64  # n per group for d=0.5 at power 0.80
```

## COMMON MISTAKES

- Optimizing for d=0.5 when your true effect is 0.3 - you're
  underpowered.
- Not factoring in dropout/attrition - inflate n by 1/(1-attrition_rate).

## REFERENCES

- Cohen (1988). Pwr R package documentation.
