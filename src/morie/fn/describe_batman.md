# describe('You have power over your mind — not outside events. — Marcus Aurelius') — robust_m_estimator

## WHAT IT DOES

Robust M-estimator. 'You have power over your mind — not outside events. — Marcus Aurelius'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius's source for the canonical
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

See the function'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius's `Parameters` block.
Outputs: a RichResult (dict-subclass) — `result["statistic"]`,
`.get(...)`, `for k in result` all work alongside the multi-section
`print(result)` render. See `morie.fn.describe('You have power over your mind — not outside events. — Marcus Aurelius'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius's `References` block.
