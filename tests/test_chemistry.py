import pytest

from hydromcp.chemistry.molecules import molecule_properties, similarity
from hydromcp.chemistry.stoichiometry import check_balance, h2_demand, h2_for_feed


@pytest.mark.parametrize("pathway,n", [("dbt_dds", 2), ("dbt_hyd", 5), ("quinoline_hdn", 7),
                                       ("carbazole_hdn", 8), ("thiophene_hds", 4),
                                       ("naphthalene_to_decalin", 5)])
def test_pathway_h2(pathway, n):
    assert h2_demand(pathway=pathway)["h2_mol_per_mol"] == n


def test_properties():
    p = molecule_properties("4,6-dimethyldibenzothiophene")
    assert p["formula"] == "C14H12S"
    assert p["aromatic_ring_count"] == 3
    assert molecule_properties("c1ccccc1")["double_bond_equivalents"] == 4


def test_balance_and_errors():
    ok = check_balance([{"species": "dibenzothiophene"}, {"species": "hydrogen", "coeff": 2}],
                       [{"species": "biphenyl"}, {"species": "h2s"}])
    assert ok["balanced"]
    with pytest.raises(ValueError):
        h2_demand("dibenzothiophene", ["biphenyl"])  # S does not balance
    with pytest.raises(ValueError):
        molecule_properties("not_a_molecule((")


def test_feed_h2_and_similarity():
    r = h2_for_feed(0.86, s_removed_wppm=10000)
    # 0.86*350.17 lb/bbl * 1% S / 32.06 * 2 mol H2 * 379.49 scf/lbmol
    assert r["hds_scf_bbl"] == pytest.approx(0.86 * 350.17 * 0.01 / 32.06 * 2 * 379.49)
    assert similarity("dibenzothiophene", "dibenzothiophene") == pytest.approx(1.0)
