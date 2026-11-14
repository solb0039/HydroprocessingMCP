"""Catalyst loading calculations: layer depths, volumes, masses, outage and order quantities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from hydromcp.core.library import Library, get_library
from hydromcp.core.schemas import BedLoading, ProcessUnit, Proposal
from hydromcp.loading.geometry import BedGeometry

ACTIVE_REACTIONS = ("hds", "hdn", "hda", "hck")
NON_ACTIVE_FUNCTIONS = ("inert", "grading")  # excluded from active volume / LHSV basis


@dataclass
class LayerResult:
    reactor_id: str
    bed_name: str
    position: int  # 1 = top layer of the bed
    catalyst_id: str
    catalyst_function: str
    loading_method: str
    top_depth_ft: float
    bottom_depth_ft: float
    depth_ft: float
    volume_ft3: float
    density_lb_ft3: float
    mass_lb: float
    void_fraction: float
    in_bottom_head: bool
    active: bool


def plan_bed(bl: BedLoading, unit: ProcessUnit, lib: Library) -> dict:
    reactor = unit.reactor(bl.reactor_id)
    bed = reactor.bed(bl.bed_name)
    geo = BedGeometry(reactor, bed)
    min_outage = bl.outage_ft if bl.outage_ft is not None else bed.min_outage_ft
    loadable = geo.volume_between(min_outage, geo.max_depth_ft)
    warnings: list[str] = []

    results: list[LayerResult] = []
    z_bottom = geo.max_depth_ft
    n = len(bl.layers)
    for i, layer in enumerate(reversed(bl.layers)):
        cat = lib.catalyst(layer.catalyst_id)
        rho = cat.density(layer.loading_method)
        if layer.depth_ft is not None:
            z_top = z_bottom - layer.depth_ft
            vol = (geo.volume_between(max(z_top, 0.0), z_bottom)
                   + max(-z_top, 0.0) * geo.area)
        else:
            if layer.volume_ft3 is not None:
                vol = layer.volume_ft3
            elif layer.mass_lb is not None:
                vol = layer.mass_lb / rho
            else:
                vol = layer.bed_fraction * loadable
            z_top = geo.depth_for_volume_above(z_bottom, vol) if z_bottom > 0 else -vol / geo.area
        results.append(LayerResult(
            reactor_id=reactor.id, bed_name=bed.name, position=n - i,
            catalyst_id=cat.id, catalyst_function=cat.function,
            loading_method=layer.loading_method, top_depth_ft=round(z_top, 4),
            bottom_depth_ft=round(z_bottom, 4), depth_ft=round(z_bottom - z_top, 4),
            volume_ft3=vol, density_lb_ft3=rho, mass_lb=vol * rho,
            void_fraction=cat.void_fraction(layer.loading_method),
            in_bottom_head=z_bottom > geo.h_cyl + 1e-9,
            active=(cat.function not in NON_ACTIVE_FUNCTIONS
                    and any(cat.rva(r) > 0 for r in ACTIVE_REACTIONS)),
        ))
        z_bottom = z_top
    results.reverse()
    outage = z_bottom
    if outage < 0:
        warnings.append(f"Bed {reactor.id}/{bed.name} overfilled by {-outage:.2f} ft")
    elif outage + 1e-9 < min_outage:
        warnings.append(f"Bed {reactor.id}/{bed.name} outage {outage:.2f} ft is below the "
                        f"minimum {min_outage:.2f} ft")
    for r in results:
        cat = lib.catalyst(r.catalyst_id)
        if r.in_bottom_head and r.active:
            warnings.append(f"Active catalyst {r.catalyst_id} extends into the bottom head of "
                            f"{reactor.id}/{bed.name}; usually only support material goes there")
        if r.loading_method == "dense" and cat.density_dense_lb_ft3 is None:
            warnings.append(f"{cat.id}: no dense density given; assumed sock x 1.12")
    return {
        "reactor_id": reactor.id,
        "bed_name": bed.name,
        "bed_total_volume_ft3": geo.total_volume_ft3,
        "bed_loadable_volume_ft3": loadable,
        "outage_ft": outage,
        "loaded_volume_ft3": sum(r.volume_ft3 for r in results),
        "active_volume_ft3": sum(r.volume_ft3 for r in results if r.active),
        "loaded_mass_lb": sum(r.mass_lb for r in results),
        "active_mass_lb": sum(r.mass_lb for r in results if r.active),
        "layers": results,
        "warnings": warnings,
    }


def plan_loading(proposal: Proposal, unit: ProcessUnit | None = None,
                 lib: Library | None = None, contingency_pct: float = 5.0) -> dict:
    """Full loading plan for a proposal: per-bed layers, totals and catalyst order summary."""
    lib = lib or get_library()
    unit = unit or lib.unit(proposal.unit_id)
    beds = [plan_bed(bl, unit, lib) for bl in proposal.beds]
    layers = [lr for b in beds for lr in b["layers"]]
    order: dict[str, dict] = {}
    for lr in layers:
        o = order.setdefault(lr.catalyst_id, {"catalyst_id": lr.catalyst_id, "volume_ft3": 0.0,
                                              "mass_lb": 0.0})
        o["volume_ft3"] += lr.volume_ft3
        o["mass_lb"] += lr.mass_lb
    for cid, o in order.items():
        cat = lib.catalyst(cid)
        o["order_mass_lb"] = o["mass_lb"] * (1 + contingency_pct / 100.0)
        if cat.package_size_lb:
            o["packages"] = math.ceil(o["order_mass_lb"] / cat.package_size_lb)
            o["package_size_lb"] = cat.package_size_lb
        if cat.cost_per_lb is not None:
            o["estimated_cost"] = o["order_mass_lb"] * cat.cost_per_lb
    return {
        "proposal_id": proposal.id,
        "unit_id": unit.id,
        "beds": [{**{k: v for k, v in b.items() if k != "layers"},
                  "layers": [asdict(lr) for lr in b["layers"]]} for b in beds],
        "totals": {
            "loaded_volume_ft3": sum(b["loaded_volume_ft3"] for b in beds),
            "active_volume_ft3": sum(b["active_volume_ft3"] for b in beds),
            "loaded_mass_lb": sum(b["loaded_mass_lb"] for b in beds),
            "active_mass_lb": sum(b["active_mass_lb"] for b in beds),
        },
        "order_summary": list(order.values()),
        "contingency_pct": contingency_pct,
        "warnings": [w for b in beds for w in b["warnings"]],
        "_layers": layers,
    }


def public(plan: dict) -> dict:
    """Strip internal objects from a plan for JSON output."""
    return {k: v for k, v in plan.items() if not k.startswith("_")}
