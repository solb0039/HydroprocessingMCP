"""In-memory session store for datasets, fitted models and calibrations.

Large objects (DataFrames, fitted models) stay server-side and are referenced by id so
MCP tool results stay small.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class DatasetEntry:
    id: str
    df: pd.DataFrame
    source: str
    created: datetime = field(default_factory=datetime.now)
    units: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        df = self.df
        idx = df.index
        info: dict[str, Any] = {
            "dataset_id": self.id,
            "source": self.source,
            "rows": len(df),
            "columns": list(map(str, df.columns)),
        }
        if isinstance(idx, pd.DatetimeIndex) and len(idx):
            info["start"] = idx.min().isoformat()
            info["end"] = idx.max().isoformat()
            if len(idx) > 1:
                info["median_interval"] = str(pd.Series(idx).diff().median())
        if self.units:
            info["units"] = self.units
        if self.notes:
            info["notes"] = self.notes
        return info


class SessionStore:
    def __init__(self) -> None:
        self.datasets: dict[str, DatasetEntry] = {}
        self.models: dict[str, Any] = {}
        self.calibrations: dict[str, Any] = {}
        self._counter = itertools.count(1)

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._counter)}"

    def add_dataset(self, df: pd.DataFrame, source: str, units: dict | None = None,
                    dataset_id: str | None = None) -> DatasetEntry:
        did = dataset_id or self.new_id("ds")
        entry = DatasetEntry(id=did, df=df, source=source, units=dict(units or {}))
        self.datasets[did] = entry
        return entry

    def dataset(self, dataset_id: str) -> DatasetEntry:
        try:
            return self.datasets[dataset_id]
        except KeyError:
            raise KeyError(
                f"Unknown dataset '{dataset_id}'. Loaded: {sorted(self.datasets)}"
            ) from None

    def add_model(self, model: Any, prefix: str = "model") -> str:
        mid = self.new_id(prefix)
        self.models[mid] = model
        return mid

    def model(self, model_id: str) -> Any:
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"Unknown model '{model_id}'. Available: {sorted(self.models)}") from None

    def add_calibration(self, cal: Any) -> str:
        cid = self.new_id("cal")
        self.calibrations[cid] = cal
        return cid

    def calibration(self, cal_id: str) -> Any:
        try:
            return self.calibrations[cal_id]
        except KeyError:
            raise KeyError(
                f"Unknown calibration '{cal_id}'. Available: {sorted(self.calibrations)}"
            ) from None


STORE = SessionStore()
