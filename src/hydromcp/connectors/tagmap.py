"""Tag maps rename raw historian/LIMS tags to canonical variable names (and apply
scale/offset), so equations can be applied with consistent column names.

YAML format (<data dir>/tagmaps/<name>.yaml):
    tags:
      - {tag: "11TI101.PV", name: bed1_t_in, unit: degF, description: Bed 1 inlet}
      - {tag: "11FI100.PV", name: feed_rate_bpd, unit: BPD}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from hydromcp import config
from hydromcp.core.schemas import TagMapping


def load_tagmap(name_or_path: str) -> list[TagMapping]:
    p = Path(name_or_path).expanduser()
    if not p.exists():
        p = config.data_dir() / "tagmaps" / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Tag map '{name_or_path}' not found")
    doc = yaml.safe_load(p.read_text()) or {}
    return [TagMapping.model_validate(t) for t in doc.get("tags", [])]


def apply_tagmap(df: pd.DataFrame, mappings: list[TagMapping],
                 keep_unmapped: bool = True) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    rename, units = {}, {}
    for m in mappings:
        if m.tag in out.columns:
            if m.scale != 1.0 or m.offset != 0.0:
                out[m.tag] = out[m.tag] * m.scale + m.offset
            rename[m.tag] = m.name
            if m.unit:
                units[m.name] = m.unit
    out = out.rename(columns=rename)
    if not keep_unmapped:
        out = out[[c for c in out.columns if c in rename.values()]]
    return out, units
