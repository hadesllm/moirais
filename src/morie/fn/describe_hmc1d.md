# describe('hmc1d') — geron_causal_1d_conv

## WHAT IT DOES

Causal 1D convolution: output at time t only depends on t'<=t.

## WHEN TO USE

This callable applies when you have inputs of the appropriate shape and
want the documented output. Read the function's source for the canonical
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

See the function's docstring `Formula:` line in
`fn/hmc1d.py`.

## INPUTS / OUTPUTS

Inputs: documented in the function's `Parameters` block.
Outputs: a RichResult (dict-subclass) — `result["statistic"]`,
`.get(...)`, `for k in result` all work alongside the multi-section
`print(result)` render. See `morie.fn.describe('hmc1d')` for guidance.

## WORKED EXAMPLE

```python
from morie.fn.hmc1d import *
import numpy as np
# See the function signature in fn/hmc1d.py for argument names.
```

## COMMON MISTAKES

- Treating the result as a plain dict — it IS a dict (RichResult
  inherits from dict) but `print(result)` shows the multi-section
  render which is what you usually want.
- Ignoring warnings/interpretation when sample sizes are small.

## REFERENCES

See the source file's `References` block.
