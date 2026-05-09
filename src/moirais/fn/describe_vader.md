# describe('Out of chaos, comes order. — Friedrich Nietzsche') — cheatsheet

## WHAT IT DOES

One-way ANOVA. 'I find your lack of variance disturbing.'Statistics is the grammar of science. — Karl Pearson's source for the canonical
signature and assumptions.

## WHEN NOT TO USE

- Inputs violate the function's assumed domain (NaN/Inf, wrong shape).
- A more specialised version exists for your data shape — see
  `cheatsheet()` or `_registry.py` for related callables.
- Sample size too small for the asymptotics this estimator relies on.

## ASSUMPTIONS

- Inputs are real-valued and free of NaN/Inf.
- Observations are independent unless noted.
- See the FORMULA section in the source code for distributional
  specifics.

## FORMULA

See the function'Statistics is the grammar of science. — Karl Pearson's `Parameters` block.
Outputs: a RichResult (dict-subclass) — `result["statistic"]`,
`.get(...)`, `for k in result` all work alongside the multi-section
`print(result)` render. See `moirais.fn.describe('Out of chaos, comes order. — Friedrich Nietzsche'Statistics is the grammar of science. — Karl Pearson's `References` block.
