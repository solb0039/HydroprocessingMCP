# User equations

Anything placed in this folder is loaded at server start (and by the
`analysis_reload_user_equations` tool). A user equation with the same name as a built-in
one **replaces** it, so you can swap in your company's definitions (e.g. your WABT or
normalized-WABT method) without touching the package code.

## Option 1 — YAML expression (simple formulas)

```yaml
# data/equations/my_equations.yaml
equations:
  - name: wabt_bed                      # overrides the built-in
    category: temperature
    description: Bed WABT, company standard (Tin + Tout)/2
    reference: Refinery tech manual 4.2
    expression: "(t_in + t_out)/2"
    inputs:
      t_in:  {unit: degF, description: Bed inlet}
      t_out: {unit: degF, description: Bed outlet}
    output: {name: wabt, unit: degF}
```

Allowed: + - * / ^ ( ), numbers, declared input names and `exp, log/ln, log10, sqrt,
abs, Min, Max, sin, cos, tan, pi`. Expressions are validated, then parsed with sympy.
An input may have a `default:` value.

Equations created through the `analysis_register_equation` tool are saved to
`user_equations.yaml` in this folder.

## Option 2 — Python plugin (anything more complex)

```python
# data/equations/my_kinetics.py
import numpy as np
from hydromcp.core.equations import equation

@equation(
    name="hds_k_langmuir",
    category="kinetics",
    inputs={
        "c_in": ("wppm", "Feed S"),
        "c_out": ("wppm", "Product S"),
        "lhsv": ("1/hr", "LHSV"),
        "h2s_pp": ("psia", "H2S partial pressure"),
        "k_h2s": ("1/psia", "H2S adsorption constant", 0.02),   # 3rd item = default
    },
    outputs={"k": ("1/hr", "Inhibition-corrected rate constant")},
    reference="Your reference",
)
def hds_k_langmuir(c_in, c_out, lhsv, h2s_pp, k_h2s=0.02):
    return lhsv * np.log(c_in / c_out) * (1 + k_h2s * h2s_pp)
```

Functions must work element-wise (use numpy functions) so they can be applied to a
whole time series with `analysis_apply_equation`. Return a single value, a tuple
(matching `outputs` order) or a dict `{output_name: value}`. Inputs declared with
`list_inputs=("temps",)` receive a list (e.g. one series per bed or thermocouple).
