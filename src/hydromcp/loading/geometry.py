"""Reactor vessel geometry: cross-section, head volumes and cumulative bed volume vs depth."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from hydromcp.core.schemas import Bed, Reactor


def head_depth_ft(diameter_ft: float, head_type: str) -> float:
    return {"2:1_elliptical": diameter_ft / 4.0, "hemispherical": diameter_ft / 2.0,
            "flat": 0.0}[head_type]


def head_volume_ft3(diameter_ft: float, head_type: str) -> float:
    """Full volume of one head (2:1 elliptical = pi D^3/24, hemispherical = pi D^3/12)."""
    return {"2:1_elliptical": math.pi * diameter_ft ** 3 / 24.0,
            "hemispherical": math.pi * diameter_ft ** 3 / 12.0, "flat": 0.0}[head_type]


def head_cap_volume_ft3(diameter_ft: float, head_type: str, h_from_bottom_ft: float) -> float:
    """Volume of a bottom head filled to height h above its lowest point (spheroidal head)."""
    a = head_depth_ft(diameter_ft, head_type)
    if a == 0:
        return 0.0
    h = min(max(h_from_bottom_ft, 0.0), a)
    r = diameter_ft / 2.0
    return math.pi * r * r * (h * h / a - h ** 3 / (3.0 * a * a))


def vessel_volume(diameter_ft: float, tangent_length_ft: float,
                  head_type: str = "2:1_elliptical") -> dict:
    """Total volume of a vertical vessel: cylinder between tangent lines plus two heads."""
    cyl = math.pi * diameter_ft ** 2 / 4.0 * tangent_length_ft
    head = head_volume_ft3(diameter_ft, head_type)
    return {"cross_section_ft2": math.pi * diameter_ft ** 2 / 4.0, "cylinder_ft3": cyl,
            "head_each_ft3": head, "head_depth_ft": head_depth_ft(diameter_ft, head_type),
            "total_ft3": cyl + 2 * head}


class BedGeometry:
    """Cumulative loadable volume V(z) for depth z measured down from the top of the bed space.

    Below the straight side, if the bed includes the bottom head, the cross-section follows
    the head shape. Vertical internals area is subtracted along the full depth."""

    def __init__(self, reactor: Reactor, bed: Bed) -> None:
        self.d = reactor.inside_diameter_ft
        self.head_type = reactor.head_type
        self.bed = bed
        self.area_gross = math.pi * self.d ** 2 / 4.0
        self.area = self.area_gross - bed.internals_area_ft2
        if self.area <= 0:
            raise ValueError(f"Bed '{bed.name}' internals area exceeds vessel cross-section")
        self.h_cyl = bed.cylindrical_height_ft
        self.a = head_depth_ft(self.d, self.head_type) if bed.includes_bottom_head else 0.0

    @property
    def max_depth_ft(self) -> float:
        return self.h_cyl + self.a

    def volume_to_depth(self, z: float) -> float:
        """Volume between the top of the bed space and depth z."""
        z = min(max(z, 0.0), self.max_depth_ft)
        if z <= self.h_cyl:
            return self.area * z
        d = z - self.h_cyl  # depth below the bottom tangent line
        full = head_volume_ft3(self.d, self.head_type)
        cap_below = head_cap_volume_ft3(self.d, self.head_type, self.a - d)
        internals = self.bed.internals_area_ft2 * d
        return self.area * self.h_cyl + max(full - cap_below - internals, 0.0)

    def volume_between(self, z_top: float, z_bottom: float) -> float:
        return self.volume_to_depth(z_bottom) - self.volume_to_depth(z_top)

    @property
    def total_volume_ft3(self) -> float:
        return self.volume_to_depth(self.max_depth_ft)

    def depth_for_volume_above(self, z_bottom: float, volume: float) -> float:
        """Top depth of a layer of given volume whose bottom sits at z_bottom."""
        avail = self.volume_to_depth(z_bottom)
        if volume >= avail:
            # overfilled: extend linearly above the bed top (negative depth) for reporting
            return -(volume - avail) / self.area
        return brentq(lambda z: self.volume_between(z, z_bottom) - volume, 0.0, z_bottom,
                      xtol=1e-6)
