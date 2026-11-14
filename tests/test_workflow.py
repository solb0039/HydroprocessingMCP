"""End-to-end workflow through the MCP server, on synthetic data generated from the model
with a known 1.5 F/month deactivation rate."""

import pytest


@pytest.fixture(scope="module")
def ds(call, sample_csv):
    s = call("analysis_load_file", path=sample_csv, tagmap_name="example", dataset_id="hist")
    assert s["rows"] == 2520
    return s


def test_load_and_tagmap(ds):
    assert "feed_rate_bpd" in ds["columns"] and "bed1_t_in" in ds["columns"]
    assert ds["units"]["feed_rate_bpd"] == "BPD"


def test_section1_calcs(call, ds):
    call("analysis_apply_equation", dataset_id="hist", equation="wabt_bed",
         column_map={"t_in": "bed1_t_in", "t_out": "bed1_t_out"}, output_prefix="bed1_")
    call("analysis_apply_equation", dataset_id="hist", equation="wabt_bed",
         column_map={"t_in": "bed2_t_in", "t_out": "bed2_t_out"}, output_prefix="bed2_")
    plan = call("loading_plan_reactor_loading", proposal_id="ULSD-1:current")
    w = [b["active_mass_lb"] for b in plan["beds"]]
    r = call("analysis_apply_equation", dataset_id="hist", equation="wabt_reactor",
             column_map={"bed_wabts": ["bed1_wabt", "bed2_wabt"], "bed_catalyst_weights": w})
    assert r["new_columns"] == ["wabt"]
    call("analysis_apply_equation", dataset_id="hist", equation="lhsv",
         column_map={"feed_rate_bpd": "feed_rate_bpd",
                     "catalyst_volume_ft3": plan["totals"]["active_volume_ft3"]})
    call("analysis_apply_equation", dataset_id="hist", equation="h2_partial_pressure",
         column_map={"pressure_psig": "reactor_p_psig", "h2_mol_frac": "recycle_h2_mol"},
         output_prefix="")
    h2 = call("analysis_apply_equation", dataset_id="hist", equation="h2_consumption",
              column_map={"makeup_mscfd": "makeup_mscfd", "makeup_purity": "makeup_purity",
                          "feed_rate_bpd": "feed_rate_bpd", "purge_mscfd": "purge_mscfd",
                          "purge_purity": "recycle_h2_mol", "solution_loss_mscfd": 400})
    assert 280 < h2["statistics"]["h2_consumption_scf_bbl"]["mean"] < 345
    n = call("analysis_apply_equation", dataset_id="hist", equation="normalized_wabt",
             column_map={"wabt": "wabt", "lhsv": "lhsv", "c_in": "feed_s_wppm",
                         "c_out": "prod_s_wppm", "h2_pp": "h2_pp_psia", "lhsv_ref": 1.287,
                         "c_in_ref": 12000, "c_out_ref": 8, "h2_pp_ref": 900})
    assert n["new_columns"] == ["wabt_norm"]
    age = call("analysis_add_catalyst_age", dataset_id="hist", feed_rate_column="feed_rate_bpd",
               catalyst_mass_lb=sum(w))
    assert age["latest"]["days_on_stream"] == pytest.approx(419.83, abs=0.1)
    ss = call("analysis_detect_steady_state", dataset_id="hist",
              columns=["feed_rate_bpd", "wabt"], window="24h",
              tolerances={"feed_rate_bpd": 600, "wabt": 3.0})
    assert 0 < ss["steady_fraction"] <= 1


def test_section3_deactivation_and_eor(call, ds):
    fit = call("forecast_fit_deactivation", dataset_id="hist", y_column="wabt_norm",
               x_column="days_on_stream", kind="linear")
    # model deactivation 1.5 F/month; normalization uses fixed exponents so allow tolerance
    rate_month = fit["params"]["b"] * 30.4375
    assert rate_month == pytest.approx(1.5, abs=0.25)
    eor = call("forecast_predict_end_of_run", model_id=fit["model_id"], max_wabt_f=760)
    assert eor["days_remaining"] > 0 and "eor_date" in eor
    cmp = call("forecast_compare_deactivation_models", dataset_id="hist")
    assert cmp["best"]
    fc = call("forecast_forecast_variable", dataset_id="hist", column="reactor_dp_psi",
              horizon=60, method="linear")
    assert fc["end_of_horizon"]["forecast"] > fc["last_observed"]["value"] - 3


def test_section4_calibrate_predict_compare(call, ds):
    cal = call("proposals_calibrate_from_dataset", unit_id="ULSD-1", dataset_id="hist",
               start="2025-12-01", end="2026-02-20",
               column_map={"wabt_f": "wabt", "feed_rate_bpd": "feed_rate_bpd",
                           "feed_sulfur_wppm": "feed_s_wppm", "feed_sg": "feed_sg",
                           "product_sulfur_wppm": "prod_s_wppm",
                           "h2_partial_pressure_psia": "h2_pp_psia"})
    assert cal["age_months"] > 10
    # data were generated with k_ref(hds) from the design basis -> calibration recovers it
    from hydromcp.core.library import get_library
    from hydromcp.proposals.predict import calibrate

    lib = get_library()
    u = lib.unit("ULSD-1")
    k_true = calibrate(u, lib.proposal("ULSD-1:current"), u.design_basis).k_ref["hds"]
    assert cal["k_ref"]["hds"] == pytest.approx(k_true, rel=0.12)

    cmp = call("proposals_compare_proposals",
               proposal_ids=["ULSD-1:current", "PROP-COMO-HA", "PROP-STACKED"],
               calibration_id=cal["calibration_id"], product_sulfur_target_wppm=8)
    rows = {r["proposal_id"]: r for r in cmp["table"]}
    assert rows["PROP-COMO-HA"]["delta_wabt_f"] < -5
    assert cmp["ranking_by_cycle_then_wabt"][0] == "PROP-COMO-HA"

    wi = call("forecast_what_if_scenario", calibration_id=cal["calibration_id"],
              changes={"feed_rate_bpd": 40000})
    assert wi["delta_wabt_f"] > 0


def test_chemistry_tools(call):
    r = call("chemistry_h2_demand", pathway="dbt_hyd")
    assert r["h2_mol_per_mol"] == 5
    p = call("chemistry_molecule_properties", smiles_or_name="dibenzothiophene")
    assert p["formula"] == "C12H8S"
    assert p["element_wt_pct"]["S"] == pytest.approx(17.40, abs=0.05)
