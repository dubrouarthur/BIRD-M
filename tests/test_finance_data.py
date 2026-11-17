import math

import pytest

from bird_m.data.checks import extract_numbers, numbers_check
from bird_m.data.finance_tasks import generate
from bird_m.world.finance import cash_bridge, generate_company, ratios

SEEDS = range(300)


# ------------------------------------------------------------------ BIRD-World invariants
@pytest.mark.parametrize("seed", SEEDS)
def test_accounting_invariants(seed):
    c = generate_company(seed)
    assert len(c.years) == 3 and c.employees >= 2
    for fy in c.years:
        assert fy.total_actif == fy.total_passif  # balance sheet balances
        assert fy.tresorerie >= 0 and fy.concours_bancaires >= 0
        assert min(fy.tresorerie, fy.concours_bancaires) == 0
        assert fy.rcai == (
            fy.chiffre_affaires
            - fy.achats_consommes
            - fy.charges_externes
            - fy.impots_taxes
            - fy.charges_personnel
            - fy.dotations_amortissements
            + fy.resultat_financier
        )
        assert fy.resultat_net == fy.rcai + fy.resultat_exceptionnel - fy.impot_societes
        assert fy.impot_societes == round(
            max(0, fy.rcai + fy.resultat_exceptionnel) * 0.25
        )
    for prev, cur in zip(c.years, c.years[1:]):
        b = cash_bridge(prev, cur)
        assert b["variation_tresorerie_nette"] == (
            b["caf"]
            - b["variation_bfr"]
            - b["investissements"]
            + b["variation_dettes_financieres"]
            - b["dividendes"]
        )
        assert (
            cur.capitaux_propres
            == prev.capitaux_propres + cur.resultat_net - cur.dividendes
        )
        assert cur.dividendes <= max(0, prev.resultat_net)


def test_generator_is_deterministic_and_diverse():
    a, b = generate_company(42), generate_company(42)
    assert [y.to_dict() for y in a.years] == [y.to_dict() for y in b.years]
    cs = [generate_company(s) for s in range(400)]
    assert len({c.sector.key for c in cs}) >= 8
    assert {c.health for c in cs} == {"saine", "fragile", "difficulte"}
    ebe = sorted(ratios(c.years[-1])["taux_ebe"] for c in cs)
    assert 5 < ebe[len(ebe) // 2] < 15  # realistic median EBITDA margin


# ------------------------------------------------------------------ number checks
@pytest.mark.parametrize(
    "text,want",
    [
        ("18 400,5 k€", [18400.5]),
        ("18 400 k€", [18400.0]),
        ("1,234.5 and 1.234,5", [1234.5, 1234.5]),
        ("marge de 12,3 % et −4.2 %", [12.3, -4.2]),
        ("en 2025, 2 340 clients", [2025.0, 2340.0]),
        ("2024 1500", [2024.0, 1500.0]),
        ("ratio 0,6 an", [0.6]),
    ],
)
def test_extract_numbers(text, want):
    assert extract_numbers(text) == want


def test_numbers_check_rejects_wrong_figures():
    facts = {"ebe": 12.4, "ca": 18400}
    assert numbers_check("EBE 12,4 % sur un CA de 18 400 k€", facts)["score"] == 1
    bad = numbers_check("EBE 14,1 % sur un CA de 18 400 k€", facts)
    assert bad["score"] == 0.5 and bad["missing"] == ["ebe"]
    assert numbers_check("rien", {"x": float("nan")})["checked"] == 0


# ------------------------------------------------------------------ tasks
def test_tasks_exact_targets_pass_their_own_checks():
    recs = generate(600, seed=3)
    exact = [r for r in recs if r["kind"] == "exact"]
    assert len(exact) > 250
    for r in exact:
        res = numbers_check(r["target"], r["facts"])
        assert res["score"] == 1, (r["id"], res)


def test_tasks_schema_and_languages():
    recs = generate(300, seed=5)
    assert {r["lang"] for r in recs} == {"fr", "en"}
    assert {r["kind"] for r in recs} == {"exact", "teacher"}
    for r in recs:
        assert r["messages"][0]["role"] == "user" and r["messages"][0]["content"]
        assert (r["target"] is None) == (r["kind"] == "teacher")
        assert all(
            not (isinstance(v, float) and math.isnan(v)) for v in r["facts"].values()
        )
    fr = next(r for r in recs if r["lang"] == "fr")
    assert (
        "k€" in fr["messages"][0]["content"]
        and "Chiffre d'affaires" in fr["messages"][0]["content"]
    )


def test_company_offset_gives_disjoint_splits():
    train = {r["company"]["id"] for r in generate(200, seed=0, company_offset=0)}
    bench = {r["company"]["id"] for r in generate(200, seed=0, company_offset=900_000)}
    assert not train & bench
