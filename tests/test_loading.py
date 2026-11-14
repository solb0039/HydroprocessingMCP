import math

import pytest

from hydromcp.core.library import get_library
from hydromcp.core.schemas import Bed, BedLoading, Layer, Proposal, Reactor
from hydromcp.loading import geometry
from hydromcp.loading.hydraulics import ergun_dp_per_length
from hydromcp.loading.loading import plan_loading


def test_vessel_and_heads():
    v = geometry.vessel_volume(12, 40)
    assert v["cylinder_ft3"] == pytest.approx(math.pi * 36 * 40)
    assert v["head_each_ft3"] == pytest.approx(math.pi * 12 ** 3 / 24)
    assert geometry.head_cap_volume_ft3(12, "2:1_elliptical", 3.0) == pytest.approx(v["head_each_ft3"])
    hemi = geometry.head_cap_volume_ft3(12, "hemispherical", 6.0)
    assert hemi == pytest.approx(2 / 3 * math.pi * 6 ** 3)


def test_bed_geometry_inverse():
    r = Reactor(id="R", inside_diameter_ft=10, beds=[Bed(name="b", cylindrical_height_ft=20,
                                                         includes_bottom_head=True)])
    g = geometry.BedGeometry(r, r.beds[0])
    assert g.max_depth_ft == 22.5
    assert g.total_volume_ft3 == pytest.approx(math.pi * 25 * 20 + math.pi * 1000 / 24)
    z_top = g.depth_for_volume_above(22.5, 100.0)
    assert g.volume_between(z_top, 22.5) == pytest.approx(100.0, rel=1e-6)


def test_current_loading_plan():
    plan = plan_loading(get_library().proposal("ULSD-1:current"))
    bed1 = plan["beds"][0]
    area = math.pi * 144 / 4 - 0.15
    como = [l for l in bed1["layers"] if l["catalyst_id"] == "CAT-COMO-REF"][0]
    assert como["volume_ft3"] == pytest.approx(15 * area)
    assert como["mass_lb"] == pytest.approx(15 * area * 48)
    assert bed1["outage_ft"] == pytest.approx(0.75)
    bed2 = plan["beds"][1]
    assert bed2["layers"][-1]["in_bottom_head"] is True
    assert plan["warnings"] == []
    o = {x["catalyst_id"]: x for x in plan["order_summary"]}
    assert o["CAT-COMO-REF"]["packages"] == math.ceil(o["CAT-COMO-REF"]["order_mass_lb"] / 2000)


def test_overfill_and_fraction_layers():
    lib = get_library()
    p = Proposal(id="t", unit_id="ULSD-1", beds=[BedLoading(
        reactor_id="R-101", bed_name="bed1",
        layers=[Layer(catalyst_id="CAT-COMO-REF", depth_ft=19.0)])])
    assert any("overfilled" in w for w in plan_loading(p, lib=lib)["warnings"])
    p2 = Proposal(id="t2", unit_id="ULSD-1", beds=[BedLoading(
        reactor_id="R-101", bed_name="bed1",
        layers=[Layer(catalyst_id="CAT-COMO-REF", bed_fraction=0.5),
                Layer(catalyst_id="CAT-NIMO-HA", mass_lb=20000)])])
    plan = plan_loading(p2, lib=lib)
    l1, l2 = plan["beds"][0]["layers"]
    assert l2["volume_ft3"] == pytest.approx(20000 / 47)
    assert l1["volume_ft3"] == pytest.approx(0.5 * plan["beds"][0]["bed_loadable_volume_ft3"])


def test_layer_requires_one_size():
    with pytest.raises(ValueError):
        Layer(catalyst_id="x", depth_ft=1, volume_ft3=2)


def test_ergun_laminar_limit():
    # pure viscous term check: very low flow -> dP proportional to flow
    a = ergun_dp_per_length(1.0, 50, 1.0, 1.0, 1.5, 0.4)
    b = ergun_dp_per_length(2.0, 50, 1.0, 1.0, 1.5, 0.4)
    assert b / a == pytest.approx(2.0, rel=1e-3)
