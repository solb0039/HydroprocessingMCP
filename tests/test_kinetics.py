import math

import pytest

from hydromcp.core.library import get_library
from hydromcp.core.schemas import Catalyst, OperatingBasis
from hydromcp.proposals.kinetics import LayerModel, SystemModel
from hydromcp.proposals.predict import calibrate, predict


def basis(**kw):
    d = dict(feed_rate_bpd=24000, feed_sulfur_wppm=10000, h2_partial_pressure_psia=1000)
    d.update(kw)
    return OperatingBasis(**d)


def cat(**kw):
    d = dict(id="c", density_sock_lb_ft3=45, activity={"hds": 1.0, "hdn": 1.0},
             reaction_order={"hds": 1.0})
    d.update(kw)
    return Catalyst(**d)


def test_first_order_layers_in_series_equal_single_bed():
    # LHSV 1 overall (5614.583 ft3 at 24,000 BPD); split into two layers should not matter
    one = SystemModel([LayerModel("a", cat(), 5614.583)], {"hds": 2.0}, h2pp_ref_psia=1000)
    two = SystemModel([LayerModel("a", cat(), 2000.0), LayerModel("b", cat(), 3614.583)],
                      {"hds": 2.0}, h2pp_ref_psia=1000)
    b = basis()
    assert one.outlet("hds", 10000, 700, b) == pytest.approx(10000 * math.exp(-2.0))
    assert two.outlet("hds", 10000, 700, b) == pytest.approx(one.outlet("hds", 10000, 700, b))


def test_nth_order_and_inverse():
    m = SystemModel([LayerModel("a", cat(reaction_order={"hds": 1.5}), 5614.583)], {"hds": 0.5},
                    h2pp_ref_psia=1000)
    b = basis()
    c = m.outlet("hds", 10000, 700, b)
    assert c == pytest.approx((10000 ** -0.5 + 0.5 * 0.5) ** -2)
    t = m.required_wabt("hds", 10000, 10.0, b)
    assert m.outlet("hds", 10000, t, b) == pytest.approx(10.0, rel=1e-3)


def test_deactivation_as_temperature_loss():
    m = SystemModel([LayerModel("a", cat(deactivation_rate_f_per_month=2.0), 5000)], {"hds": 2.0})
    b = basis(h2_partial_pressure_psia=1000)
    m.h2pp_ref_psia = 1000
    t0 = m.required_wabt("hds", 10000, 10, b, 0)
    t6 = m.required_wabt("hds", 10000, 10, b, 6)
    assert t6 - t0 == pytest.approx(12.0, abs=0.05)


def test_calibration_round_trip_and_cycle():
    lib = get_library()
    u = lib.unit("ULSD-1")
    cal = calibrate(u, lib.proposal("ULSD-1:current"), u.design_basis)
    r = predict(lib.proposal("ULSD-1:current"), cal, u.design_basis, u, lib, 8.0)
    assert r["wabt_f"] == pytest.approx(672.0, abs=0.05)
    assert r["product_nitrogen_wppm"] == pytest.approx(12.0, rel=1e-3)
    assert r["cycle_length_months"] == pytest.approx((760 - 672) / 1.5, abs=0.3)
    assert r["yield_total_wt_pct"] == pytest.approx(100 + r["h2_consumption_wt_pct"])


def test_calibration_with_age_gives_fresh_basis():
    lib = get_library()
    u = lib.unit("ULSD-1")
    b = u.design_basis.model_copy(update={"wabt_f": 672 + 1.5 * 12})
    cal = calibrate(u, lib.proposal("ULSD-1:current"), b, age_months=12)
    fresh = calibrate(u, lib.proposal("ULSD-1:current"), u.design_basis)
    # grading layers age faster (2.0-2.2 F/mo) than the main catalyst, so not exact
    assert cal.k_ref["hds"] == pytest.approx(fresh.k_ref["hds"], rel=5e-3)


def test_hck_yields():
    lib = get_library()
    hck = lib.catalyst("CAT-HCK-DIST")
    m = SystemModel([LayerModel("a", lib.catalyst("CAT-COMO-REF"), 3000),
                     LayerModel("b", hck, 3000)], {"hds": 2.0, "hck": 0.5}, h2pp_ref_psia=1500)
    from hydromcp.proposals.predict import DEFAULT_H2_STOICH, _h2_and_yields

    b = basis(feed_conversion_fraction_wt=80, h2_partial_pressure_psia=1500)
    heavy = m.outlet("hck", 80, 740, b)
    y = _h2_and_yields(m, b, 20, None, None, heavy, DEFAULT_H2_STOICH)
    assert set(hck.yield_selectivity) <= set(y["yields_wt_pct_of_feed"])
    assert y["yield_total_wt_pct"] == pytest.approx(100 + y["h2_consumption_wt_pct"])
