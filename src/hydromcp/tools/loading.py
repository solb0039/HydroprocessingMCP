"""Section 2 MCP tools: reactor geometry and catalyst loading calculations."""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from hydromcp.core.equations import to_jsonable
from hydromcp.core.library import get_library
from hydromcp.core.schemas import ProcessUnit, Proposal
from hydromcp.loading import geometry, hydraulics
from hydromcp.loading.loading import plan_loading, public

mcp = FastMCP("loading")


def _proposal(proposal_id: str | None, proposal: Proposal | None) -> Proposal:
    if proposal is not None:
        return proposal
    if not proposal_id:
        raise ValueError("Give proposal_id (or '<unit_id>:current') or a proposal object")
    return get_library().proposal(proposal_id)


@mcp.tool
def vessel_volume(inside_diameter_ft: float, tangent_length_ft: float,
                  head_type: Literal["2:1_elliptical", "hemispherical", "flat"] = "2:1_elliptical"
                  ) -> dict:
    """Volume of a vertical reactor vessel: cylinder between tangent lines plus two heads."""
    return geometry.vessel_volume(inside_diameter_ft, tangent_length_ft, head_type)


@mcp.tool
def list_units() -> list[dict]:
    """List process units (reactors and beds) defined in data/units."""
    return [{"id": u.id, "name": u.name, "service": u.service,
             "reactors": [{"id": r.id, "inside_diameter_ft": r.inside_diameter_ft,
                           "beds": [b.name for b in r.beds]} for r in u.reactors],
             "has_current_loading": u.current_loading is not None,
             "max_wabt_f": u.max_wabt_f} for u in get_library().units()]


@mcp.tool
def get_unit(unit_id: str) -> dict:
    """Full definition of a process unit, including current loading and design basis."""
    return get_library().unit(unit_id).model_dump(exclude_none=True)


@mcp.tool
def add_unit(unit: ProcessUnit, persist: bool = True) -> dict:
    """Add or replace a process unit definition (reactors, beds, current loading)."""
    get_library().add_unit(unit, persist)
    return {"added": unit.id, "persisted": persist}


@mcp.tool
def bed_capacity(unit_id: str, reactor_id: str, bed_name: str, from_depth_ft: float = 0.0,
                 to_depth_ft: float | None = None) -> dict:
    """Loadable volume of a bed between two depths (measured down from the top of the bed
    space), including the bottom head if the bed includes it."""
    unit = get_library().unit(unit_id)
    reactor = unit.reactor(reactor_id)
    geo = geometry.BedGeometry(reactor, reactor.bed(bed_name))
    z2 = geo.max_depth_ft if to_depth_ft is None else to_depth_ft
    return {"volume_ft3": geo.volume_between(from_depth_ft, z2), "max_depth_ft": geo.max_depth_ft,
            "net_cross_section_ft2": geo.area, "volume_per_ft_ft3": geo.area,
            "bed_total_volume_ft3": geo.total_volume_ft3}


@mcp.tool
def layer_quantity(catalyst_id: str, loading_method: Literal["sock", "dense"] = "sock",
                   volume_ft3: float | None = None, depth_ft: float | None = None,
                   inside_diameter_ft: float | None = None, mass_lb: float | None = None) -> dict:
    """Quick conversion between volume, depth (in a cylinder of given ID) and mass for a
    catalyst at its sock or dense loaded density."""
    import math

    cat = get_library().catalyst(catalyst_id)
    rho = cat.density(loading_method)
    area = math.pi * inside_diameter_ft ** 2 / 4 if inside_diameter_ft else None
    if volume_ft3 is None:
        if mass_lb is not None:
            volume_ft3 = mass_lb / rho
        elif depth_ft is not None and area:
            volume_ft3 = depth_ft * area
        else:
            raise ValueError("Give volume_ft3, mass_lb, or depth_ft with inside_diameter_ft")
    out = {"catalyst_id": cat.id, "loading_method": loading_method, "density_lb_ft3": rho,
           "volume_ft3": volume_ft3, "mass_lb": volume_ft3 * rho}
    if area:
        out["depth_ft"] = volume_ft3 / area
    if cat.package_size_lb:
        out["packages"] = math.ceil(out["mass_lb"] / cat.package_size_lb)
    return out


@mcp.tool
def plan_reactor_loading(proposal_id: str | None = None, proposal: Proposal | None = None,
                         contingency_pct: float = 5.0) -> dict:
    """Complete loading plan for a catalyst system: each layer's top/bottom depth, volume,
    mass, outage per bed, active catalyst totals, order quantities (with contingency and
    package count) and warnings (overfill, low outage, active catalyst in head).

    proposal_id may be a saved proposal or '<unit_id>:current' for the existing loading."""
    return to_jsonable(public(plan_loading(_proposal(proposal_id, proposal),
                                           contingency_pct=contingency_pct)))


@mcp.tool
def loading_pressure_drop(liquid_lb_hr: float, gas_lb_hr: float,
                          proposal_id: str | None = None, proposal: Proposal | None = None,
                          liquid_density_lb_ft3: float = 45.0, liquid_viscosity_cp: float = 0.25,
                          gas_density_lb_ft3: float = 1.2, gas_viscosity_cp: float = 0.016) -> dict:
    """Clean-bed pressure drop of each layer and bed (Ergun, gas + liquid single-phase
    terms). A screening index for comparing loadings/grading, not a trickle-bed design."""
    lib = get_library()
    prop = _proposal(proposal_id, proposal)
    plan = plan_loading(prop)
    unit = lib.unit(prop.unit_id)
    layers, beds = [], {}
    for lr in plan["_layers"]:
        cat = lib.catalyst(lr.catalyst_id)
        size = cat.particle_diameter_mm or hydraulics.equivalent_diameter_mm(
            cat.size_mm or 1.3, cat.shape)
        area = hydraulics.bed_area_ft2(unit.reactor(lr.reactor_id).inside_diameter_ft)
        dp = hydraulics.layer_pressure_drop(
            max(lr.depth_ft, 0.0), area, size, lr.void_fraction, liquid_lb_hr,
            liquid_density_lb_ft3, liquid_viscosity_cp, gas_lb_hr, gas_density_lb_ft3,
            gas_viscosity_cp)
        layers.append({"reactor_id": lr.reactor_id, "bed_name": lr.bed_name,
                       "catalyst_id": lr.catalyst_id, "depth_ft": lr.depth_ft,
                       "equivalent_diameter_mm": size, "void_fraction": lr.void_fraction, **dp})
        key = f"{lr.reactor_id}/{lr.bed_name}"
        beds[key] = beds.get(key, 0.0) + dp["dp_total_psi"]
    return to_jsonable({"layers": layers, "bed_dp_psi": beds, "total_dp_psi": sum(beds.values())})
