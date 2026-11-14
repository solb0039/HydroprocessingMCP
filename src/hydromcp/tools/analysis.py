"""Section 1 MCP tools: load historical process data and run hydroprocessing calculations."""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import FastMCP

from hydromcp import config
from hydromcp.analysis import timeseries_ops as ops
from hydromcp.connectors import files, tagmap
from hydromcp.core.equations import (
    REGISTRY,
    load_user_equations,
    make_expression_equation,
    save_expression_equation,
    to_jsonable,
)
from hydromcp.core.state import STORE

mcp = FastMCP("analysis")


def _store(df, source, map_name=None, dataset_id=None):
    units = {}
    if map_name:
        df, units = tagmap.apply_tagmap(df, tagmap.load_tagmap(map_name))
    entry = STORE.add_dataset(df, source, units, dataset_id)
    return entry.summary()


# -- data loading ---------------------------------------------------------------


@mcp.tool
def load_file(path: str, time_column: str | None = None, sheet: str | int | None = 0,
              skiprows: int | None = None, tagmap_name: str | None = None,
              dataset_id: str | None = None) -> dict:
    """Load a CSV, Excel or Parquet file of time-series process data into the session.

    time_column is auto-detected when omitted. tagmap_name applies a tag map from
    data/tagmaps to rename raw tags to canonical names. Returns a dataset summary with
    the dataset_id used by the other tools."""
    df = files.load_file(path, time_column, sheet, skiprows)
    return _store(df, f"file:{path}", tagmap_name, dataset_id)


@mcp.tool
def list_excel_sheets(path: str) -> list[str]:
    """List the sheet names in an Excel workbook."""
    return files.list_sheets(path)


@mcp.tool
def query_sql(connection: str, query: str, time_column: str | None = None,
              params: dict | None = None, tagmap_name: str | None = None,
              dataset_id: str | None = None) -> dict:
    """Run a read-only SELECT against a SQL database and load the result as a dataset.

    connection: alias of an environment variable HYDROMCP_SQL_<ALIAS> holding the
    SQLAlchemy URL (preferred, keeps credentials out of chat), or a full URL."""
    from hydromcp.connectors.sql import query_sql as q

    df = q(connection, query, params, time_column)
    return _store(df, f"sql:{connection}", tagmap_name, dataset_id)


@mcp.tool
def pi_search_tags(name_filter: str, max_count: int = 100) -> list[dict]:
    """Search PI Data Archive tags by name filter (wildcards allowed, e.g. '11TI1*')."""
    from hydromcp.connectors.pi import PIClient

    return PIClient().search_tags(name_filter, max_count)


@mcp.tool
def pi_fetch(tags: list[str], start: str, end: str = "*", interval: str = "1h",
             mode: Literal["interpolated", "recorded", "summary"] = "interpolated",
             tagmap_name: str | None = None, dataset_id: str | None = None) -> dict:
    """Fetch PI tag history through PI Web API into a dataset.

    start/end accept PI time syntax ('*-180d', '*') or ISO timestamps. 'summary' returns
    time-weighted averages per interval."""
    from hydromcp.connectors.pi import PIClient

    df = PIClient().fetch(tags, start, end, interval, mode)
    return _store(df, f"pi:{len(tags)} tags", tagmap_name, dataset_id)


@mcp.tool
def apply_tag_map(dataset_id: str, tagmap_name: str, keep_unmapped: bool = True) -> dict:
    """Rename a dataset's raw tag columns to canonical names using a tag map."""
    entry = STORE.dataset(dataset_id)
    entry.df, units = tagmap.apply_tagmap(entry.df, tagmap.load_tagmap(tagmap_name), keep_unmapped)
    entry.units.update(units)
    return entry.summary()


# -- inspection --------------------------------------------------------------------


@mcp.tool
def list_datasets() -> list[dict]:
    """List datasets loaded in this session."""
    return [e.summary() for e in STORE.datasets.values()]


@mcp.tool
def describe_dataset(dataset_id: str, columns: list[str] | None = None) -> dict:
    """Summary statistics (count, mean, std, min, quartiles, max, missing) per column."""
    entry = STORE.dataset(dataset_id)
    return {"summary": entry.summary(), "statistics": ops.describe(entry.df, columns)}


@mcp.tool
def preview_rows(dataset_id: str, n: int = 10, columns: list[str] | None = None,
                 tail: bool = False) -> dict:
    """Return the first (or last) n rows of a dataset."""
    df = STORE.dataset(dataset_id).df
    df = df[columns] if columns else df
    sub = df.tail(n) if tail else df.head(n)
    import json

    return {"rows": json.loads(sub.reset_index().to_json(orient="records", date_format="iso"))}


@mcp.tool
def window_statistics(dataset_id: str, columns: list[str], start: str | None = None,
                      end: str | None = None) -> dict:
    """Mean/median/std of columns over a time window (e.g. a test run or steady period)."""
    return ops.window_stats(STORE.dataset(dataset_id).df, columns, start, end)


@mcp.tool
def correlation_matrix(dataset_id: str, columns: list[str]) -> dict:
    """Pearson correlation matrix between columns."""
    return ops.correlate(STORE.dataset(dataset_id).df, columns)


# -- cleaning -----------------------------------------------------------------------


@mcp.tool
def clean_dataset(dataset_id: str, resample_rule: str | None = None,
                  resample_how: Literal["mean", "median", "min", "max", "last"] = "mean",
                  outlier_columns: list[str] | None = None, outlier_window: int = 25,
                  outlier_n_mad: float = 4.0, start: str | None = None, end: str | None = None,
                  drop_columns: list[str] | None = None, new_dataset_id: str | None = None) -> dict:
    """Create a cleaned copy: trim to [start, end], drop columns, remove outliers (rolling
    median +/- n MAD) and resample (e.g. '1h', '1D'). Returns the new dataset summary."""
    entry = STORE.dataset(dataset_id)
    df = entry.df
    if start or end:
        df = df.loc[start:end]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    removed = {}
    if outlier_columns:
        df, removed = ops.filter_outliers(df, outlier_columns, outlier_window, outlier_n_mad)
    if resample_rule:
        df = ops.resample(df, resample_rule, resample_how)
    new = STORE.add_dataset(df, f"clean:{dataset_id}", entry.units, new_dataset_id)
    if removed:
        new.notes.append(f"outliers removed: {removed}")
    return new.summary()


@mcp.tool
def detect_steady_state(dataset_id: str, columns: list[str], window: str = "24h",
                        tolerances: dict[str, float] | None = None, rel_tol: float = 0.02,
                        min_duration: str | None = "24h", flag_column: str = "steady_state",
                        max_windows: int = 50) -> dict:
    """Find steady-state periods where the rolling std of every listed column is within
    tolerance (absolute per column, else rel_tol x rolling mean). Adds a boolean column."""
    entry = STORE.dataset(dataset_id)
    flag, windows = ops.detect_steady_state(entry.df, columns, window, tolerances, rel_tol,
                                            min_duration)
    entry.df[flag_column] = flag
    return {"steady_fraction": float(flag.mean()), "n_windows": len(windows),
            "windows": windows[:max_windows], "flag_column": flag_column}


@mcp.tool
def add_catalyst_age(dataset_id: str, feed_rate_column: str, catalyst_mass_lb: float | None = None,
                     start_of_run: str | None = None) -> dict:
    """Add days_on_stream, cumulative_feed_bbl and (if catalyst mass given) catalyst_life_bbl_lb
    columns, measured from start_of_run (default first timestamp)."""
    entry = STORE.dataset(dataset_id)
    df = entry.df
    df["days_on_stream"] = ops.days_on_stream(df, start_of_run)
    df["cumulative_feed_bbl"] = ops.cumulative_feed(df, feed_rate_column, start_of_run)
    cols = ["days_on_stream", "cumulative_feed_bbl"]
    if catalyst_mass_lb:
        df["catalyst_life_bbl_lb"] = df["cumulative_feed_bbl"] / catalyst_mass_lb
        cols.append("catalyst_life_bbl_lb")
    entry.notes.append(f"catalyst age from {start_of_run or df.index[0]}")
    last = df[cols].dropna().iloc[-1].to_dict() if len(df) else {}
    return {"added_columns": cols, "latest": to_jsonable(last)}


# -- equations ----------------------------------------------------------------------


@mcp.tool
def list_equations(category: str | None = None) -> list[dict]:
    """List available hydroprocessing equations (built-in and user) with inputs/units."""
    return [{"name": e.name, "category": e.category, "description": e.description,
             "inputs": [f"{v.name}{'[]' if v.is_list else ''} ({v.unit})" for v in e.inputs],
             "outputs": [f"{v.name} ({v.unit})" for v in e.outputs], "source": e.source}
            for e in REGISTRY.list(category)]


@mcp.tool
def get_equation(name: str) -> dict:
    """Full definition of one equation: inputs, units, defaults, reference."""
    return REGISTRY.get(name).to_dict()


@mcp.tool
def run_equation(name: str, inputs: dict[str, Any]) -> dict:
    """Evaluate an equation for single values, e.g. run_equation('wabt_bed',
    {'t_in': 650, 't_out': 690}). List inputs take a list of numbers."""
    return to_jsonable(REGISTRY.evaluate(name, **inputs))


@mcp.tool
def apply_equation(dataset_id: str, equation: str, column_map: dict[str, Any],
                   output_prefix: str | None = None) -> dict:
    """Apply an equation to every row of a dataset and append the result column(s).

    column_map maps each equation input to a column name, a list of column names (for list
    inputs such as per-bed temperatures) or a constant number, e.g.
    {'t_in': 'bed1_tin', 't_out': 'bed1_tout'}. Returns stats of the new columns."""
    entry = STORE.dataset(dataset_id)
    entry.df, new_cols = ops.apply_equation(entry.df, equation, column_map, output_prefix)
    eq = REGISTRY.get(equation)
    for v in eq.outputs:
        col = f"{output_prefix or ''}{v.name}"
        if v.unit:
            entry.units[col] = v.unit
    return {"dataset_id": dataset_id, "new_columns": new_cols,
            "statistics": ops.describe(entry.df, new_cols)}


@mcp.tool
def register_equation(name: str, expression: str, inputs: dict[str, dict], output: dict,
                      category: str = "user", description: str = "", reference: str = "",
                      persist: bool = True) -> dict:
    """Define a new equation (or override a built-in) from a math expression.

    inputs: {'t_in': {'unit': 'degF', 'description': '...', 'default': 0}}
    output: {'name': 'wabt', 'unit': 'degF'}
    Allowed functions: exp, log/ln, log10, sqrt, abs, Min, Max, sin, cos, tan, pi.
    persist=True saves it to data/equations/user_equations.yaml."""
    eq = make_expression_equation(name, expression, inputs, output, category, description,
                                  reference)
    REGISTRY.register(eq)
    if persist:
        save_expression_equation(eq, config.data_dir() / "equations" / "user_equations.yaml")
    return eq.to_dict()


@mcp.tool
def reload_user_equations() -> dict:
    """Reload user equation plugins (*.py) and YAML expressions from data/equations."""
    return {"loaded": load_user_equations(config.data_dir() / "equations")}


@mcp.tool
def export_dataset(dataset_id: str, filename: str | None = None,
                   columns: list[str] | None = None) -> dict:
    """Write a dataset (with calculated columns) to CSV in the output folder."""
    entry = STORE.dataset(dataset_id)
    df = entry.df[columns] if columns else entry.df
    path = config.output_dir() / (filename or f"{dataset_id}.csv")
    df.to_csv(path)
    return {"path": str(path), "rows": len(df), "columns": list(map(str, df.columns))}
