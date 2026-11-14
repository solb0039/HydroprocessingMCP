"""File-backed library of catalysts, process units and proposals (YAML/JSON in the data dir).

  <data dir>/catalysts/*.yaml   -> {catalysts: [Catalyst, ...]}
  <data dir>/units/*.yaml       -> {unit: ProcessUnit}  (or a bare ProcessUnit mapping)
  <data dir>/proposals/*.yaml   -> {proposal: Proposal}
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hydromcp import config
from hydromcp.core.schemas import Catalyst, ProcessUnit, Proposal


def _read(path: Path):
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _files(folder: Path):
    if not folder.is_dir():
        return []
    return sorted([*folder.glob("*.yaml"), *folder.glob("*.yml"), *folder.glob("*.json")])


class Library:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._catalysts: dict[str, Catalyst] = {}
        self._catalyst_files: dict[str, Path] = {}
        self._units: dict[str, ProcessUnit] = {}
        self._proposals: dict[str, Proposal] = {}
        self.errors: list[str] = []
        self.reload()

    @property
    def base(self) -> Path:
        return self.root or config.data_dir()

    # -- loading -----------------------------------------------------------
    def reload(self) -> None:
        self._catalysts.clear()
        self._units.clear()
        self._proposals.clear()
        self.errors = []
        for path in _files(self.base / "catalysts"):
            try:
                doc = _read(path) or {}
                for item in doc.get("catalysts", []):
                    cat = Catalyst.model_validate(item)
                    self._catalysts[cat.id] = cat
                    self._catalyst_files[cat.id] = path
            except Exception as exc:
                self.errors.append(f"{path.name}: {exc}")
        for path in _files(self.base / "units"):
            try:
                doc = _read(path) or {}
                unit = ProcessUnit.model_validate(doc.get("unit", doc))
                self._units[unit.id] = unit
            except Exception as exc:
                self.errors.append(f"{path.name}: {exc}")
        for path in _files(self.base / "proposals"):
            try:
                doc = _read(path) or {}
                prop = Proposal.model_validate(doc.get("proposal", doc))
                self._proposals[prop.id] = prop
            except Exception as exc:
                self.errors.append(f"{path.name}: {exc}")

    # -- catalysts -----------------------------------------------------------
    def catalysts(self) -> list[Catalyst]:
        return list(self._catalysts.values())

    def catalyst(self, cid: str) -> Catalyst:
        try:
            return self._catalysts[cid]
        except KeyError:
            raise KeyError(f"Unknown catalyst '{cid}'. Library: {sorted(self._catalysts)}") from None

    def add_catalyst(self, cat: Catalyst, persist: bool = True,
                     filename: str = "user_catalysts.yaml") -> None:
        self._catalysts[cat.id] = cat
        if persist:
            path = self.base / "catalysts" / filename
            doc = (_read(path) if path.exists() else None) or {"catalysts": []}
            items = [c for c in doc.get("catalysts", []) if c.get("id") != cat.id]
            items.append(cat.model_dump(exclude_none=True))
            doc["catalysts"] = items
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(doc, sort_keys=False))

    # -- units -----------------------------------------------------------
    def units(self) -> list[ProcessUnit]:
        return list(self._units.values())

    def unit(self, uid: str) -> ProcessUnit:
        try:
            return self._units[uid]
        except KeyError:
            raise KeyError(f"Unknown unit '{uid}'. Library: {sorted(self._units)}") from None

    def add_unit(self, unit: ProcessUnit, persist: bool = True) -> None:
        self._units[unit.id] = unit
        if persist:
            path = self.base / "units" / f"{unit.id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump({"unit": unit.model_dump(exclude_none=True)},
                                           sort_keys=False))

    # -- proposals -----------------------------------------------------------
    def proposals(self, unit_id: str | None = None) -> list[Proposal]:
        return [p for p in self._proposals.values() if unit_id is None or p.unit_id == unit_id]

    def proposal(self, pid: str) -> Proposal:
        if pid in self._proposals:
            return self._proposals[pid]
        # allow "current" loading of a unit as "<unit_id>:current"
        if pid.endswith(":current"):
            unit = self.unit(pid.split(":")[0])
            if unit.current_loading is None:
                raise KeyError(f"Unit '{unit.id}' has no current_loading defined.")
            return unit.current_loading
        raise KeyError(f"Unknown proposal '{pid}'. Library: {sorted(self._proposals)}")

    def add_proposal(self, prop: Proposal, persist: bool = True) -> None:
        self._proposals[prop.id] = prop
        if persist:
            path = self.base / "proposals" / f"{prop.id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump({"proposal": prop.model_dump(exclude_none=True)},
                                           sort_keys=False))


_LIBRARY: Library | None = None


def get_library() -> Library:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = Library()
    return _LIBRARY


def set_library(lib: Library) -> None:
    global _LIBRARY
    _LIBRARY = lib
