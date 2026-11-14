"""OSIsoft/AVEVA PI Web API connector.

Configure with PI_WEB_API_URL plus either PI_WEB_API_USER/PI_WEB_API_PASSWORD (basic)
or PI_WEB_API_TOKEN (bearer). PI_DATA_SERVER names the default Data Archive.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pandas as pd

from hydromcp import config


class PIClient:
    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None,
                 data_server: str | None = None) -> None:
        s = config.pi_settings()
        self.base_url = (base_url or s["base_url"]).rstrip("/")
        if not self.base_url:
            raise ValueError("PI Web API URL not configured. Set PI_WEB_API_URL.")
        self.data_server = data_server or s["data_server"]
        if client is None:
            auth = (s["user"], s["password"]) if s["user"] else None
            headers = {"Authorization": f"Bearer {s['token']}"} if s["token"] else {}
            client = httpx.Client(auth=auth, headers=headers, verify=s["verify"], timeout=60.0)
        self.http = client
        self._webids: dict[str, str] = {}

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.http.get(f"{self.base_url}{path}", params=params)
        r.raise_for_status()
        return r.json()

    def _tag_path(self, tag: str) -> str:
        if tag.startswith("\\\\"):
            return tag
        if not self.data_server:
            raise ValueError(f"Tag '{tag}' has no server; use \\\\SERVER\\TAG or set PI_DATA_SERVER")
        return f"\\\\{self.data_server}\\{tag}"

    def webid(self, tag: str) -> str:
        if tag not in self._webids:
            data = self._get("/points", {"path": self._tag_path(tag)})
            self._webids[tag] = data["WebId"]
        return self._webids[tag]

    def search_tags(self, name_filter: str, max_count: int = 100) -> list[dict]:
        if not self.data_server:
            raise ValueError("Set PI_DATA_SERVER to search tags")
        server = self._get("/dataservers", {"path": f"\\\\{self.data_server}"})
        data = self._get(f"/dataservers/{server['WebId']}/points",
                         {"nameFilter": name_filter, "maxCount": max_count})
        return [{"name": i.get("Name"), "descriptor": i.get("Descriptor"),
                 "units": i.get("EngineeringUnits"), "point_type": i.get("PointType")}
                for i in data.get("Items", [])]

    @staticmethod
    def _value(v: Any) -> float:
        if isinstance(v, dict):  # digital state / system state, e.g. {"Name": "Bad Input"}
            v = v.get("Value")
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    def fetch(self, tags: list[str], start: str, end: str, interval: str | None = "1h",
              mode: str = "interpolated", summary_type: str = "Average") -> pd.DataFrame:
        """mode: 'interpolated' (needs interval), 'recorded', or 'summary' (time-weighted
        averages per interval). PI time syntax is accepted, e.g. start='*-90d', end='*'."""
        frames = []
        for tag in tags:
            wid = self.webid(tag)
            if mode == "interpolated":
                data = self._get(f"/streams/{wid}/interpolated",
                                 {"startTime": start, "endTime": end, "interval": interval})
                items = data.get("Items", [])
            elif mode == "recorded":
                data = self._get(f"/streams/{wid}/recorded",
                                 {"startTime": start, "endTime": end, "maxCount": 150000})
                items = data.get("Items", [])
            elif mode == "summary":
                data = self._get(f"/streams/{wid}/summary",
                                 {"startTime": start, "endTime": end, "summaryDuration": interval,
                                  "summaryType": summary_type, "calculationBasis": "TimeWeighted"})
                items = [i["Value"] for i in data.get("Items", [])]
            else:
                raise ValueError("mode must be interpolated, recorded or summary")
            idx = pd.to_datetime([i["Timestamp"] for i in items], utc=True)
            vals = [self._value(i.get("Value")) for i in items]
            frames.append(pd.Series(vals, index=idx, name=tag))
        df = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
        df.index.name = "timestamp"
        return df
