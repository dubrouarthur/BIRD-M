import re

from bird_medium import finance_fr as F


def num(s: str) -> float:
    return float(s.replace(" ", "").replace(" ", "").replace("€", "").replace(",", "."))


def test_formats_francais():
    assert F.eur(1234.5) == "1 234,50 €"
    assert F.pct(5.5) == "5,5 %"
    assert F.pct(20) == "20 %"


def test_tva_ttc_exact():
    for r in F.generate(80, seed=3):
        a = r["messages"][-1]["content"]
        m = re.search(r"Montant HT : ([\d  ,]+) €\n- TVA \(([\d,]+) %\) : ([\d  ,]+) €", a)
        if m:
            ht, t, tva = num(m.group(1)), num(m.group(2)), num(m.group(3))
            assert abs(tva - round(ht * t / 100, 2)) < 0.011


def test_tous_les_exemples_ont_un_raisonnement():
    recs = F.generate(16, seed=0)
    assert {r["source"] for r in recs} == {"finance_fr"}
    assert all(r["thinking"] and r["messages"][-1]["reasoning_content"] for r in recs)
    assert len({r["messages"][0]["content"] for r in recs}) == 16


def test_deterministe():
    assert F.generate(10, seed=1) == F.generate(10, seed=1)
