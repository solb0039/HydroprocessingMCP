"""Runtime configuration, read from environment variables.

HYDROMCP_DATA_DIR      Folder holding catalysts/, reactors/, equations/, tagmaps/, proposals/
                       (default: <repo>/data when running from source, else ./data)
HYDROMCP_OUTPUT_DIR    Where exports are written (default: <data dir>/output)
PI_WEB_API_URL         e.g. https://piserver/piwebapi
PI_WEB_API_USER / PI_WEB_API_PASSWORD   Basic auth credentials (optional)
PI_WEB_API_TOKEN       Bearer token (optional, alternative to basic auth)
PI_WEB_API_VERIFY_SSL  "false" to disable TLS verification (default true)
PI_DATA_SERVER         Default PI Data Archive name used to resolve bare tag names
HYDROMCP_SQL_<ALIAS>   SQLAlchemy URL for a named SQL connection, e.g. HYDROMCP_SQL_LIMS
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    env = os.environ.get("HYDROMCP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if _REPO_DATA.is_dir():
        return _REPO_DATA
    return Path.cwd() / "data"


def output_dir() -> Path:
    env = os.environ.get("HYDROMCP_OUTPUT_DIR")
    path = Path(env).expanduser().resolve() if env else data_dir() / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sql_url(alias_or_url: str) -> str:
    """Resolve a named connection (HYDROMCP_SQL_<ALIAS>) or pass a URL through."""
    if "://" in alias_or_url:
        return alias_or_url
    key = f"HYDROMCP_SQL_{alias_or_url.upper()}"
    url = os.environ.get(key)
    if not url:
        raise ValueError(f"No SQL connection named '{alias_or_url}'. Set environment variable {key}.")
    return url


def pi_settings() -> dict:
    return {
        "base_url": os.environ.get("PI_WEB_API_URL", "").rstrip("/"),
        "user": os.environ.get("PI_WEB_API_USER"),
        "password": os.environ.get("PI_WEB_API_PASSWORD"),
        "token": os.environ.get("PI_WEB_API_TOKEN"),
        "verify": os.environ.get("PI_WEB_API_VERIFY_SSL", "true").lower() != "false",
        "data_server": os.environ.get("PI_DATA_SERVER"),
    }
