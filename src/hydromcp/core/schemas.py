"""Pydantic data models for catalysts, reactors, loadings (proposals) and unit operating bases.

Units are US field units unless the field name says otherwise (see hydromcp.core.units).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Reaction = Literal["hds", "hdn", "hda", "hck"]
REACTIONS: tuple[str, ...] = ("hds", "hdn", "hda", "hck")

DEFAULT_ORDER = {"hds": 1.5, "hdn": 1.0, "hda": 1.0, "hck": 1.0}
DEFAULT_EA_KJ = {"hds": 110.0, "hdn": 125.0, "hda": 80.0, "hck": 160.0}
DEFAULT_H2PP_EXP = {"hds": 0.5, "hdn": 1.0, "hda": 1.0, "hck": 0.5}


# ---------------------------------------------------------------------------
# Catalyst
# ---------------------------------------------------------------------------


class Catalyst(BaseModel):
    """A catalyst (or inert/grading material) with its performance values.

    activity: relative volume activity (RVA) per reaction versus the reference catalyst
      used to calibrate the unit (reference RVA = 1.0). Missing reaction => 0 (inactive).
    activation_energy_kj_mol / reaction_order / h2pp_exponent: kinetic parameters per
      reaction; missing entries use library defaults.
    deactivation_rate_f_per_month: WABT increase needed per month to hold constant
      performance at design severity (vendor-style).
    yield_selectivity: for hydrocracking catalysts, fraction of converted material that
      goes to each product cut (normalised to 1).
    """

    id: str
    name: str | None = None
    vendor: str | None = None
    function: Literal[
        "inert", "grading", "hdm", "hds", "hdn", "pretreat", "hck", "aromatics_sat", "dewax"
    ] = "hds"
    metals: str | None = Field(None, description="e.g. CoMo, NiMo, NiW, NiCoMo")
    shape: str | None = Field(None, description="e.g. cylinder, trilobe, quadlobe, sphere, ring")
    size_mm: float | None = Field(None, description="Nominal size (diameter) in mm")
    particle_diameter_mm: float | None = Field(
        None, description="Equivalent sphere diameter for pressure drop (defaults to size_mm)"
    )
    density_sock_lb_ft3: float = Field(..., gt=0, description="Sock/loose loaded density")
    density_dense_lb_ft3: float | None = Field(None, gt=0, description="Dense loaded density")
    void_fraction_sock: float = Field(0.40, gt=0, lt=1)
    void_fraction_dense: float = Field(0.36, gt=0, lt=1)
    activity: dict[Reaction, float] = Field(default_factory=dict)
    activation_energy_kj_mol: dict[Reaction, float] = Field(default_factory=dict)
    reaction_order: dict[Reaction, float] = Field(default_factory=dict)
    h2pp_exponent: dict[Reaction, float] = Field(default_factory=dict)
    deactivation_rate_f_per_month: float = Field(0.0, ge=0)
    h2_consumption_factor: float = Field(1.0, gt=0, description="Multiplier on chemical H2 use")
    max_temperature_f: float | None = None
    yield_selectivity: dict[str, float] | None = None
    package_size_lb: float | None = Field(None, gt=0, description="Drum/supersack net weight")
    cost_per_lb: float | None = None
    notes: str | None = None

    @field_validator("yield_selectivity")
    @classmethod
    def _norm_selectivity(cls, v):
        if v:
            total = sum(v.values())
            if total <= 0:
                raise ValueError("yield_selectivity must sum to a positive number")
            return {k: val / total for k, val in v.items()}
        return v

    def density(self, method: str) -> float:
        if method == "dense":
            if self.density_dense_lb_ft3 is None:
                # typical dense/sock uplift when vendor value missing
                return self.density_sock_lb_ft3 * 1.12
            return self.density_dense_lb_ft3
        return self.density_sock_lb_ft3

    def void_fraction(self, method: str) -> float:
        return self.void_fraction_dense if method == "dense" else self.void_fraction_sock

    def rva(self, reaction: str) -> float:
        return float(self.activity.get(reaction, 0.0))

    def ea(self, reaction: str) -> float:
        return float(self.activation_energy_kj_mol.get(reaction, DEFAULT_EA_KJ[reaction]))

    def order(self, reaction: str) -> float:
        return float(self.reaction_order.get(reaction, DEFAULT_ORDER[reaction]))

    def h2pp_exp(self, reaction: str) -> float:
        return float(self.h2pp_exponent.get(reaction, DEFAULT_H2PP_EXP[reaction]))


# ---------------------------------------------------------------------------
# Reactor geometry
# ---------------------------------------------------------------------------


class Bed(BaseModel):
    """A catalyst bed. Depth is measured downward from the top of the bed space
    (e.g. from the underside of the distributor tray/quench deck).

    cylindrical_height_ft: straight-side height available for loading.
    includes_bottom_head: True for the lowest bed when material is also loaded into the
      reactor bottom head (usually support balls).
    internals_area_ft2: cross-section occupied by vertical internals (thermowells,
      downcomers) subtracted from loaded volume.
    """

    name: str
    cylindrical_height_ft: float = Field(..., gt=0)
    includes_bottom_head: bool = False
    internals_area_ft2: float = Field(0.0, ge=0)
    min_outage_ft: float = Field(0.0, ge=0, description="Minimum free space above catalyst")
    thermocouple_elevations_ft: list[float] | None = None


class Reactor(BaseModel):
    id: str
    name: str | None = None
    inside_diameter_ft: float = Field(..., gt=0)
    head_type: Literal["2:1_elliptical", "hemispherical", "flat"] = "2:1_elliptical"
    beds: list[Bed]
    design_pressure_psig: float | None = None
    max_wabt_f: float | None = None

    def bed(self, name: str) -> Bed:
        for b in self.beds:
            if b.name == name:
                return b
        raise KeyError(f"Reactor '{self.id}' has no bed '{name}'. Beds: {[b.name for b in self.beds]}")


# ---------------------------------------------------------------------------
# Loading / proposal
# ---------------------------------------------------------------------------


class Layer(BaseModel):
    """One layer of material in a bed, listed top-to-bottom. Specify exactly one of
    depth_ft, volume_ft3, mass_lb or bed_fraction (fraction of the bed's loadable volume)."""

    catalyst_id: str
    loading_method: Literal["sock", "dense"] = "sock"
    depth_ft: float | None = Field(None, gt=0)
    volume_ft3: float | None = Field(None, gt=0)
    mass_lb: float | None = Field(None, gt=0)
    bed_fraction: float | None = Field(None, gt=0, le=1)

    @model_validator(mode="after")
    def _one_size(self):
        given = [x for x in (self.depth_ft, self.volume_ft3, self.mass_lb, self.bed_fraction)
                 if x is not None]
        if len(given) != 1:
            raise ValueError("Layer needs exactly one of depth_ft, volume_ft3, mass_lb, bed_fraction")
        return self


class BedLoading(BaseModel):
    reactor_id: str
    bed_name: str
    outage_ft: float | None = Field(None, ge=0, description="Free space above top layer")
    layers: list[Layer]


class Proposal(BaseModel):
    """A catalyst system design for a unit: which catalyst, how much and where.

    beds must be listed in process flow order (first reactor top bed first)."""

    id: str
    name: str | None = None
    unit_id: str
    beds: list[BedLoading]
    notes: str | None = None


# ---------------------------------------------------------------------------
# Process unit and operating basis
# ---------------------------------------------------------------------------


class OperatingBasis(BaseModel):
    """Feed and operating conditions used for kinetics predictions."""

    feed_rate_bpd: float = Field(..., gt=0)
    feed_sg: float = Field(0.86, gt=0)
    feed_sulfur_wppm: float = Field(..., gt=0)
    feed_nitrogen_wppm: float | None = Field(None, gt=0)
    feed_aromatics_wt: float | None = Field(None, ge=0, le=100, description="Total aromatics wt%")
    feed_conversion_fraction_wt: float | None = Field(
        None, ge=0, le=100, description="wt% of feed boiling above the conversion cut point (HCK)"
    )
    h2_partial_pressure_psia: float = Field(..., gt=0)
    treat_gas_ratio_scf_bbl: float | None = None
    wabt_f: float | None = Field(None, description="Actual or assumed WABT")
    product_sulfur_wppm: float | None = Field(None, gt=0)
    product_nitrogen_wppm: float | None = Field(None, gt=0)
    product_aromatics_wt: float | None = Field(None, ge=0, le=100)
    conversion_wt: float | None = Field(None, ge=0, le=100, description="HCK conversion wt%")
    bed_temperature_offsets_f: dict[str, float] | None = Field(
        None, description="Per-bed temperature offset from WABT, keyed 'reactor_id/bed_name'"
    )


class ProcessUnit(BaseModel):
    id: str
    name: str | None = None
    service: str | None = Field(None, description="e.g. ULSD, VGO pretreat, hydrocracker")
    reactors: list[Reactor]
    current_loading: Proposal | None = None
    design_basis: OperatingBasis | None = None
    max_wabt_f: float = 780.0
    product_sulfur_spec_wppm: float | None = None
    product_cuts: list[str] | None = Field(
        None, description="Product names for yield predictions, e.g. naphtha, kero, diesel"
    )

    def reactor(self, rid: str) -> Reactor:
        for r in self.reactors:
            if r.id == rid:
                return r
        raise KeyError(f"Unit '{self.id}' has no reactor '{rid}'.")


class TagMapping(BaseModel):
    """Map a raw historian/LIMS tag to a canonical variable name and unit."""

    tag: str
    name: str
    unit: str | None = None
    description: str | None = None
    scale: float = 1.0
    offset: float = 0.0
