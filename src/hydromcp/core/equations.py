"""Pluggable equation registry.

Every hydroprocessing calculation (built-in or user supplied) is an `Equation` with
named, unit-tagged inputs and outputs. Equations accept scalars, numpy arrays or pandas
Series, so the same equation evaluates a single point or a whole time series.

Adding your own equations
-------------------------
1. Python plugin: drop a ``*.py`` file into ``<data dir>/equations/`` ::

       from hydromcp.core.equations import equation

       @equation(
           name="my_wabt",
           category="analysis",
           inputs={"t_in": ("degF", "Bed inlet T"), "t_out": ("degF", "Bed outlet T")},
           outputs={"wabt": ("degF", "Weighted average bed temperature")},
           reference="Company standard XYZ",
       )
       def my_wabt(t_in, t_out):
           return (t_in + 2 * t_out) / 3

2. YAML expression: drop a ``*.yaml`` file into ``<data dir>/equations/`` ::

       equations:
         - name: my_wabt
           category: analysis
           description: Weighted average bed temperature
           expression: "(t_in + 2*t_out)/3"
           inputs:
             t_in: {unit: degF, description: Bed inlet T}
             t_out: {unit: degF, description: Bed outlet T}
           output: {name: wabt, unit: degF}

A user equation with the same name as a built-in one replaces it.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass
class Variable:
    name: str
    unit: str = ""
    description: str = ""
    default: Any = None
    is_list: bool = False  # input takes a list of values/series (e.g. one per bed)

    def to_dict(self) -> dict:
        d = {"name": self.name, "unit": self.unit, "description": self.description}
        if self.default is not None:
            d["default"] = self.default
        if self.is_list:
            d["is_list"] = True
        return d


@dataclass
class Equation:
    name: str
    category: str
    func: Callable[..., Any]
    inputs: list[Variable]
    outputs: list[Variable]
    description: str = ""
    reference: str = ""
    source: str = "builtin"
    expression: str | None = None
    tags: list[str] = field(default_factory=list)

    def input_names(self) -> list[str]:
        return [v.name for v in self.inputs]

    def evaluate(self, **kwargs) -> dict[str, Any]:
        """Evaluate with keyword inputs. Missing inputs fall back to declared defaults."""
        args = {}
        missing = []
        for var in self.inputs:
            if var.name in kwargs and kwargs[var.name] is not None:
                args[var.name] = kwargs[var.name]
            elif var.default is not None:
                args[var.name] = var.default
            else:
                missing.append(var.name)
        if missing:
            raise ValueError(f"Equation '{self.name}' missing inputs: {missing}")
        unknown = set(kwargs) - set(self.input_names())
        if unknown:
            raise ValueError(
                f"Equation '{self.name}' got unknown inputs {sorted(unknown)}; "
                f"expected {self.input_names()}"
            )
        result = self.func(**args)
        if isinstance(result, dict):
            return result
        if isinstance(result, tuple):
            return {v.name: r for v, r in zip(self.outputs, result)}
        return {self.outputs[0].name: result}

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "inputs": [v.to_dict() for v in self.inputs],
            "outputs": [v.to_dict() for v in self.outputs],
            "reference": self.reference,
            "source": self.source,
        }
        if self.expression:
            d["expression"] = self.expression
        return d


class EquationRegistry:
    def __init__(self) -> None:
        self._eqs: dict[str, Equation] = {}

    def register(self, eq: Equation) -> Equation:
        if eq.name in self._eqs:
            log.info("Equation '%s' (%s) replaces existing (%s)", eq.name, eq.source,
                     self._eqs[eq.name].source)
        self._eqs[eq.name] = eq
        return eq

    def get(self, name: str) -> Equation:
        try:
            return self._eqs[name]
        except KeyError:
            raise KeyError(
                f"Unknown equation '{name}'. Available: {sorted(self._eqs)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._eqs

    def list(self, category: str | None = None) -> list[Equation]:
        eqs = sorted(self._eqs.values(), key=lambda e: (e.category, e.name))
        return [e for e in eqs if category is None or e.category == category]

    def evaluate(self, name: str, **inputs) -> dict[str, Any]:
        return self.get(name).evaluate(**inputs)

    def remove(self, name: str) -> None:
        self._eqs.pop(name, None)


REGISTRY = EquationRegistry()


def _parse_vars(spec: dict | None) -> list[Variable]:
    out = []
    for name, val in (spec or {}).items():
        if isinstance(val, Variable):
            out.append(val)
        elif isinstance(val, dict):
            out.append(Variable(name=name, unit=val.get("unit", ""),
                                description=val.get("description", ""),
                                default=val.get("default"), is_list=val.get("is_list", False)))
        elif isinstance(val, (tuple, list)):
            unit = val[0] if len(val) > 0 else ""
            desc = val[1] if len(val) > 1 else ""
            default = val[2] if len(val) > 2 else None
            out.append(Variable(name=name, unit=unit, description=desc, default=default))
        else:
            out.append(Variable(name=name, unit=str(val)))
    return out


def equation(
    name: str,
    category: str,
    inputs: dict,
    outputs: dict,
    reference: str = "",
    description: str | None = None,
    list_inputs: tuple[str, ...] = (),
    registry: EquationRegistry | None = None,
    source: str | None = None,
):
    """Decorator that registers a function as an Equation.

    inputs / outputs: {name: (unit, description[, default])} or {name: {unit:..., ...}}.
    list_inputs: names of inputs that accept a list (one value/series per bed etc.).
    """

    def deco(func):
        ins = _parse_vars(inputs)
        for v in ins:
            if v.name in list_inputs:
                v.is_list = True
        mod = getattr(func, "__module__", "") or ""
        src = source or ("builtin" if mod.startswith("hydromcp.") else "user")
        eq = Equation(
            name=name,
            category=category,
            func=func,
            inputs=ins,
            outputs=_parse_vars(outputs),
            description=description or (func.__doc__ or "").strip().split("\n\n")[0],
            reference=reference,
            source=src,
        )
        (registry or REGISTRY).register(eq)
        func.equation = eq
        return func

    return deco


# ---------------------------------------------------------------------------
# Expression equations (YAML or registered at runtime through the MCP tool)
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "exp", "log", "ln", "log10", "sqrt", "sin", "cos", "tan", "abs", "Abs",
    "Min", "Max", "pi", "E", "Piecewise",
}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAFE_CHARS = re.compile(r"^[A-Za-z0-9_\s\.\+\-\*/\^\(\),<>=&|]*$")


def _validate_expression(expression: str, input_names: list[str]) -> None:
    if not _SAFE_CHARS.match(expression):
        raise ValueError("Expression contains disallowed characters.")
    if "__" in expression:
        raise ValueError("Expression may not contain '__'.")
    allowed = set(input_names) | _ALLOWED_FUNCS
    bad = [t for t in _TOKEN.findall(expression) if t not in allowed]
    if bad:
        raise ValueError(
            f"Expression uses unknown names {sorted(set(bad))}. Declare them as inputs "
            f"or use allowed functions {sorted(_ALLOWED_FUNCS)}."
        )


def make_expression_equation(
    name: str,
    expression: str,
    inputs: dict,
    output: dict | str,
    category: str = "user",
    description: str = "",
    reference: str = "",
) -> Equation:
    """Build an Equation from a math expression string (validated, then parsed by sympy)."""
    import sympy as sp

    ins = _parse_vars(inputs)
    names = [v.name for v in ins]
    _validate_expression(expression, names)
    symbols = {n: sp.Symbol(n) for n in names}
    local = dict(symbols)
    local.update({"ln": sp.log, "log": sp.log, "log10": lambda x: sp.log(x, 10),
                  "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
                  "Min": sp.Min, "Max": sp.Max, "pi": sp.pi, "E": sp.E,
                  "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "Piecewise": sp.Piecewise})
    expr = sp.sympify(expression.replace("^", "**"), locals=local)
    fn = sp.lambdify([symbols[n] for n in names], expr, modules="numpy")

    def func(**kw):
        return fn(*[kw[n] for n in names])

    if isinstance(output, str):
        out = [Variable(name=output)]
    else:
        out = [Variable(name=output.get("name", "result"), unit=output.get("unit", ""),
                        description=output.get("description", ""))]
    return Equation(
        name=name, category=category, func=func, inputs=ins, outputs=out,
        description=description, reference=reference, source="user", expression=str(expression),
    )


def load_user_equations(folder: Path, registry: EquationRegistry | None = None) -> list[str]:
    """Load *.py plugins and *.yaml/*.yml expression files from a folder. Returns names loaded."""
    reg = registry or REGISTRY
    loaded: list[str] = []
    if not folder.is_dir():
        return loaded
    for path in sorted(folder.glob("*.py")):
        before = set(e.name for e in reg.list())
        spec = importlib.util.spec_from_file_location(f"hydromcp_user_eq_{path.stem}", path)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:  # keep server alive on a bad plugin
                log.error("Failed to load equation plugin %s: %s", path, exc)
                continue
        loaded += sorted(set(e.name for e in reg.list()) - before)
    for path in sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml"))):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            log.error("Failed to read %s: %s", path, exc)
            continue
        for item in doc.get("equations", []):
            try:
                eq = make_expression_equation(
                    name=item["name"], expression=item["expression"],
                    inputs=item.get("inputs", {}), output=item.get("output", "result"),
                    category=item.get("category", "user"),
                    description=item.get("description", ""), reference=item.get("reference", ""),
                )
                reg.register(eq)
                loaded.append(eq.name)
            except Exception as exc:
                log.error("Bad equation %s in %s: %s", item.get("name"), path, exc)
    return loaded


def save_expression_equation(eq: Equation, path: Path) -> None:
    """Append/replace an expression equation in a YAML file so it persists across restarts."""
    doc = yaml.safe_load(path.read_text()) if path.exists() else None
    doc = doc or {"equations": []}
    items = [i for i in doc.get("equations", []) if i.get("name") != eq.name]
    items.append({
        "name": eq.name,
        "category": eq.category,
        "description": eq.description,
        "reference": eq.reference,
        "expression": eq.expression,
        "inputs": {v.name: {"unit": v.unit, "description": v.description,
                            **({"default": v.default} if v.default is not None else {})}
                   for v in eq.inputs},
        "output": {"name": eq.outputs[0].name, "unit": eq.outputs[0].unit},
    })
    doc["equations"] = items
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas results into JSON-friendly python values."""
    import pandas as pd

    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
