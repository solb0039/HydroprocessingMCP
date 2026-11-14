"""SQL connector (SQLAlchemy). Read-only: only SELECT / WITH statements are accepted.

Connections are named via environment variables HYDROMCP_SQL_<ALIAS> so credentials
never pass through the chat; a full SQLAlchemy URL is also accepted.
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, text

from hydromcp import config
from hydromcp.connectors.files import _finalize

_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|exec|execute|grant|revoke)\b",
    re.IGNORECASE,
)


def query_sql(connection: str, query: str, params: dict | None = None,
              time_column: str | None = None, max_rows: int = 1_000_000) -> pd.DataFrame:
    if not _READONLY.match(query) or _FORBIDDEN.search(query) or ";" in query.strip().rstrip(";"):
        raise ValueError("Only single read-only SELECT/WITH queries are allowed.")
    engine = create_engine(config.sql_url(connection))
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
    finally:
        engine.dispose()
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    return _finalize(df, time_column, dayfirst=False)
