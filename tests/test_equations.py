import math

import numpy as np
import pandas as pd
import pytest

from hydromcp.analysis import calcs  # noqa: F401
from hydromcp.core.equations import (
    REGISTRY,
    EquationRegistry,
    load_user_equations,
    make_expression_equation,
)


def ev(name, **kw):
    return REGISTRY.evaluate(name, **kw)


def test_wabt_and_rise():
    assert ev("wabt_bed", t_in=650, t_out=680)["wabt"] == pytest.approx(670.0)
    assert ev("wabt_bed_multipoint", temps=[650, 660, 680])["wabt"] == pytest.approx(662.5)
    assert ev("wabt_reactor", bed_wabts=[660, 690], bed_catalyst_weights=[1, 2])["wabt"] == 680


def test_space_velocity_and_h2():
    # 24,000 BPD * 5.6146 ft3/bbl / 24 h = 5614.6 ft3/h over 5614.6 ft3 -> 1.0
    assert ev("lhsv", feed_rate_bpd=24000, catalyst_volume_ft3=5614.583)["lhsv"] == pytest.approx(1.0)
    assert ev("whsv", feed_rate_bpd=24000, feed_sg=1.0,
              catalyst_mass_lb=350170)["whsv"] == pytest.approx(1.0)
    assert ev("treat_gas_ratio", treat_gas_mscfd=30000, feed_rate_bpd=20000,
              h2_purity_mol_frac=0.9)["h2_oil_ratio"] == pytest.approx(1350)
    assert ev("h2_partial_pressure", pressure_psig=985.304,
              h2_mol_frac=0.8)["h2_pp_psia"] == pytest.approx(800)
    r = ev("h2_consumption", makeup_mscfd=10000, makeup_purity=0.98, feed_rate_bpd=30000,
           purge_mscfd=1000, purge_purity=0.8)
    assert r["h2_consumption_scf_bbl"] == pytest.approx((9800 - 800) * 1000 / 30000)


def test_kinetics():
    k1 = ev("apparent_rate_constant", c_in=100, c_out=10, lhsv=2, order=1.0)["k_app"]
    assert k1 == pytest.approx(2 * math.log(10))
    k15 = ev("apparent_rate_constant", c_in=10000, c_out=10, lhsv=1, order=1.5)["k_app"]
    assert k15 == pytest.approx(2 * (10 ** -0.5 - 10000 ** -0.5))
    same = dict(wabt=680, lhsv=1.5, c_in=12000, c_out=10, h2_pp=900)
    ref = dict(lhsv_ref=1.5, c_in_ref=12000, c_out_ref=10, h2_pp_ref=900)
    assert ev("normalized_wabt", **same, **ref)["wabt_norm"] == pytest.approx(680)
    # needing more conversion at reference -> higher normalized temperature
    assert ev("normalized_wabt", **same, **{**ref, "c_out_ref": 5})["wabt_norm"] > 680
    # higher actual H2PP than reference -> normalized T higher (reference is harder)
    assert ev("normalized_wabt", **{**same, "h2_pp": 1000}, **ref)["wabt_norm"] > 680
    dt = ev("required_temperature_change", k_ratio=2.0, wabt=680, ea_kj_mol=110)["delta_t"]
    # dT ~ R T^2 ln2 / Ea = 21 K = 38 F at 680 F, 110 kJ/mol
    assert 35 < dt < 42


def test_properties():
    assert ev("api_gravity", sg=1.0)["api"] == pytest.approx(10.0)
    assert ev("specific_gravity", api=10.0)["sg"] == pytest.approx(1.0)
    # ASTM D4737: D=0.85 -> B=0; T10=215C,T50=260C,T90=310C -> CCI = 45.2
    ci = ev("cetane_index_d4737", t10_f=419, t50_f=500, t90_f=590, density_15c_g_ml=0.85)
    assert ci["cetane_index"] == pytest.approx(45.2, abs=1e-6)


def test_series_input():
    s = pd.Series([650.0, 660.0])
    out = ev("wabt_bed", t_in=s, t_out=s + 30)["wabt"]
    assert list(out) == [670.0, 680.0]


def test_missing_and_unknown_inputs():
    with pytest.raises(ValueError, match="missing"):
        ev("wabt_bed", t_in=600)
    with pytest.raises(ValueError, match="unknown"):
        ev("wabt_bed", t_in=600, t_out=610, bogus=1)


def test_expression_equation_and_safety():
    eq = make_expression_equation("my_wabt", "(t_in + 2*t_out)/3",
                                  {"t_in": {"unit": "degF"}, "t_out": {"unit": "degF"}},
                                  {"name": "wabt", "unit": "degF"})
    assert eq.evaluate(t_in=650, t_out=680)["wabt"] == pytest.approx(670)
    assert np.allclose(eq.evaluate(t_in=np.array([1.0, 2.0]), t_out=np.array([1.0, 2.0]))["wabt"],
                       [1, 2])
    eq2 = make_expression_equation("ln_test", "exp(ln(x)) + sqrt(x)", {"x": {}}, "y")
    assert eq2.evaluate(x=4.0)["y"] == pytest.approx(6.0)
    for bad in ("__import__('os')", "open('x')", "x.real", "y + 1"):
        with pytest.raises(ValueError):
            make_expression_equation("bad", bad, {"x": {}}, "y")


def test_user_plugins(tmp_path):
    # YAML override of a built-in name, loaded into a separate registry
    (tmp_path / "eqs.yaml").write_text(
        "equations:\n  - name: wabt_bed\n    expression: '(t_in + t_out)/2'\n"
        "    inputs: {t_in: {unit: degF}, t_out: {unit: degF}}\n    output: {name: wabt}\n")
    reg = EquationRegistry()
    assert load_user_equations(tmp_path, reg) == ["wabt_bed"]
    assert reg.evaluate("wabt_bed", t_in=600, t_out=700)["wabt"] == 650

    # Python plugin registering into the default registry (how real plugins work)
    plug = tmp_path / "plugins"
    plug.mkdir()
    (plug / "plug.py").write_text(
        "from hydromcp.core.equations import equation\n"
        "@equation(name='test_double', category='user', inputs={'x': ('', '')},\n"
        "          outputs={'y': ('', '')})\n"
        "def double(x):\n    return 2 * x\n")
    try:
        assert load_user_equations(plug) == ["test_double"]
        assert REGISTRY.evaluate("test_double", x=3)["y"] == 6
        assert REGISTRY.get("test_double").source == "user"
    finally:
        REGISTRY.remove("test_double")
