# priorbt - Beta-Binomial conjugate posterior update

## WHAT IT DOES

Updates a Beta(alpha, beta) prior on a probability with binomial data
(s successes in n trials), returning the Beta(alpha+s, beta+n-s)
posterior. Reports posterior mean, mode, variance.

## WHEN TO USE

- Bayesian inference for a single proportion.
- Online updating as new data arrive.
- Demonstrating how priors influence posteriors.

## WHEN NOT TO USE

- Multiple correlated proportions - use a hierarchical model.
- Continuous outcomes - use Normal-Normal or other conjugate pair.

## ASSUMPTIONS

- Trials are independent and identically distributed Bernoulli.
- Beta is a sensible prior for the success probability (almost always
  is, by virtue of being on [0, 1]).

## FORMULA

```
prior:      theta ~ Beta(alpha, beta)
likelihood: x_i ~ Bernoulli(theta), n trials, s successes
posterior:  theta|data ~ Beta(alpha + s, beta + n - s)
```
Posterior mean = (alpha+s) / (alpha+beta+n).

## INPUTS / OUTPUTS

```
priorbt(alpha_prior, beta_prior, successes, trials) -> RichResult
  returns  posterior alpha & beta, mean, mode, variance, prior mean,
           observed proportion.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import priorbt
>>> r = priorbt(1, 1, 7, 10)   # uniform prior, 7/10 successes
>>> r["post_mean"]
0.667  # ish
```

## COMMON MISTAKES

- Using Beta(0, 0) as a "Jeffreys" prior naively - the posterior is
  improper for s=0 or s=n.
- Reporting posterior mean alone - quote the credible interval too
  via `hpdint`.

## REFERENCES

- Gelman et al. (2013) Bayesian Data Analysis.
