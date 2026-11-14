"""CSV / Excel / Parquet loaders returning a DataFrame indexed by timestamp."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s.dtype)


def _finalize(df: pd.DataFrame, time_column: str | None, dayfirst: bool) -> pd.DataFrame:
    if time_column is None:
        # guess: first column that parses as datetimes
        for c in df.columns:
            if _is_text(df[c]) or "date" in str(c).lower() or "time" in str(c).lower():
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
                if parsed.notna().mean() > 0.9:
                    time_column = c
                    break
    if time_column is not None:
        if time_column not in df.columns:
            raise KeyError(f"time column '{time_column}' not found. Columns: {list(df.columns)}")
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce", dayfirst=dayfirst)
        df = df.dropna(subset=[time_column]).set_index(time_column).sort_index()
        df.index.name = "timestamp"
    # coerce numeric-looking object columns (historians often export "Bad", "I/O Timeout")
    for c in df.columns:
        if _is_text(df[c]):
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() >= 0.5 * df[c].notna().sum() and num.notna().any():
                df[c] = num
    return df


def load_file(path: str, time_column: str | None = None, sheet: str | int | None = 0,
              skiprows: int | None = None, dayfirst: bool = False) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".txt"):
        df = pd.read_csv(p, skiprows=skiprows)
    elif suffix == ".tsv":
        df = pd.read_csv(p, sep="\t", skiprows=skiprows)
    elif suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet, skiprows=skiprows)
    elif suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'")
    return _finalize(df, time_column, dayfirst)


def list_sheets(path: str) -> list[str]:
    return pd.ExcelFile(Path(path).expanduser()).sheet_names
